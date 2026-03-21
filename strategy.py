#!/usr/bin/env python3
"""
EXPERIMENT #001 - Supertrend + RSI Pullback + Dual HTF Filter (15m primary)
=====================================================================================
Hypothesis: 15m Supertrend captures intraday trends, but needs HTF confirmation to avoid
whipsaws. 4h HMA(21) defines major trend, 1h Supertrend confirms intermediate momentum.
RSI(14) pullback entries (RSI<40 in uptrend, RSI>60 in downtrend) improve entry timing.
ADX(14)>25 filters choppy regimes. This differs from Donchian by using Supertrend (ATR-based)
which adapts to volatility better than fixed Donchian channels.

Key features:
- Primary TF: 15m
- HTF filters: 4h HMA(21) + 1h Supertrend(10,3) for dual alignment
- Trend: 15m Supertrend(10,3) direction
- Entry: RSI(14) pullback (RSI<40 long, RSI>60 short) + ADX>25
- Regime: ADX percentile > 50th (avoid weakest trends)
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.20-0.30 discrete, scaled by ADX strength
- Take profit: Reduce to half at 2R profit, trail stop at 1R

Why this should beat current best:
- 15m captures more intraday opportunities than 12h
- Dual HTF filter (4h HMA + 1h Supertrend) reduces false signals
- RSI pullback entries improve risk/reward vs breakout entries
- Supertrend adapts to volatility better than fixed Donchian channels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_rsi_dualhtf_15m_1h_4h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2.0
        
        upper_band[i] = hl2 + multiplier * atr[i]
        lower_band[i] = hl2 - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            trend[i] = -1 if close[i] <= upper_band[i] else 1
        else:
            # Update upper/lower bands based on previous trend
            if trend[i - 1] == 1:
                upper_band[i] = min(upper_band[i], upper_band[i - 1])
            else:
                lower_band[i] = max(lower_band[i], lower_band[i - 1])
            
            # Determine new trend
            if close[i] <= lower_band[i]:
                supertrend[i] = upper_band[i]
                trend[i] = -1
            elif close[i] >= upper_band[i]:
                supertrend[i] = lower_band[i]
                trend[i] = 1
            else:
                supertrend[i] = supertrend[i - 1]
                trend[i] = trend[i - 1]
    
    return supertrend, trend, upper_band, lower_band


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100
    
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    
    return rsi


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(high[i] - high[i - 1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(low[i - 1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    tr_smooth = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period - 1, n):
        if tr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    dx = np.zeros(n)
    for i in range(period - 1, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx, plus_di, minus_di


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


def calculate_percentile_rank(series, window=100):
    """Calculate rolling percentile rank"""
    n = len(series)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(window - 1, n):
        if not np.isnan(series[i]):
            window_data = series[i - window + 1:i + 1]
            window_data = window_data[~np.isnan(window_data)]
            if len(window_data) > 0:
                pr[i] = np.sum(window_data <= series[i]) / len(window_data)
    
    return pr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    st_1h, trend_1h, _, _ = calculate_supertrend(
        df_1h['high'].values, 
        df_1h['low'].values, 
        df_1h['close'].values, 
        period=10, 
        multiplier=3.0
    )
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    trend_1h_aligned = align_htf_to_ltf(prices, df_1h, trend_1h)
    
    # Calculate 15m indicators
    st_15m, trend_15m, _, _ = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    atr_15m = calculate_atr(high, low, close, 14)
    rsi_15m = calculate_rsi(close, 14)
    adx_15m, plus_di_15m, minus_di_15m = calculate_adx(high, low, close, 14)
    
    # Calculate ADX percentile rank (regime filter)
    adx_pr = calculate_percentile_rank(adx_15m, 100)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.35   # Max position size with strong ADX
    MIN_SIZE = 0.15   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    entry_atr = 0.0
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(trend_1h_aligned[i]) or
            np.isnan(st_15m[i]) or np.isnan(trend_15m[i]) or
            np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or 
            np.isnan(adx_15m[i]) or np.isnan(adx_pr[i]) or
            atr_15m[i] == 0 or adx_15m[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HMA trend alignment (major trend)
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        major_trend = 1 if price_above_4h_hma else -1
        
        # 1h Supertrend direction (intermediate trend)
        intermediate_trend = int(trend_1h_aligned[i])
        
        # 15m Supertrend direction (short-term trend)
        short_term_trend = int(trend_15m[i])
        
        # ADX strength filter (only trade when ADX > 25 and in top 50th percentile)
        adx_strong = adx_15m[i] > 25
        adx_regime = adx_pr[i] > 0.50
        
        # RSI pullback signals
        rsi_oversold = rsi_15m[i] < 40  # Pullback in uptrend
        rsi_overbought = rsi_15m[i] > 60  # Pullback in downtrend
        
        # DI+ vs DI- for trend confirmation
        di_bullish = plus_di_15m[i] > minus_di_15m[i]
        di_bearish = minus_di_15m[i] > plus_di_15m[i]
        
        # Calculate position size based on ADX strength (dynamic sizing)
        adx_multiplier = min(1.0 + (adx_15m[i] - 25) / 50, 1.4)  # Max 1.4x
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * adx_multiplier))
        
        # Round to discrete levels
        if position_size < 0.20:
            position_size = 0.20
        elif position_size < 0.30:
            position_size = 0.25
        else:
            position_size = 0.30
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: All trends aligned + RSI pullback + ADX strong
        if (major_trend == 1 and intermediate_trend == 1 and short_term_trend == 1 and
            adx_strong and adx_regime and rsi_oversold and di_bullish):
            target_signal = position_size
        
        # Short entry: All trends aligned + RSI pullback + ADX strong
        elif (major_trend == -1 and intermediate_trend == -1 and short_term_trend == -1 and
              adx_strong and adx_regime and rsi_overbought and di_bearish):
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr_15m[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * entry_atr:  # 2R = 4*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr_15m[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 4.0 * entry_atr:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                entry_atr = atr_15m[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                # Exit if Supertrend reverses OR HTF alignment breaks
                supertrend_reversal_long = short_term_trend == -1
                supertrend_reversal_short = short_term_trend == 1
                hma_alignment_broken = (position_side == 1 and major_trend == -1) or \
                                       (position_side == -1 and major_trend == 1)
                
                if supertrend_reversal_long or supertrend_reversal_short or hma_alignment_broken:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = position_size * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals
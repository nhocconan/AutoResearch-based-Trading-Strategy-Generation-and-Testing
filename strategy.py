#!/usr/bin/env python3
"""
EXPERIMENT #080 - Supertrend + RSI Pullback Zone + Volume Confirmation (30m primary)
=====================================================================================
Hypothesis: 30m Supertrend captures intraday trends, but entries on pullbacks to RSI 45-55
zone (not extremes) provide better risk/reward. 4h HMA filter ensures we trade with HTF trend.
Volume confirmation (taker_buy_volume ratio > 55%) filters weak breakouts. This differs from
previous failures by using Supertrend (not KAMA) + RSI pullback ZONE (not extremes) + volume.

Key features:
- Primary TF: 30m
- HTF filter: 4h HMA(21) for trend direction
- Trend: Supertrend(10, 3) for direction
- Entry: RSI(14) pullback to 45-55 zone IN direction of Supertrend
- Volume: taker_buy_volume_ratio > 0.55 for longs, < 0.45 for shorts
- Regime: ADX(14) > 20 (avoid chop)
- Stoploss: 2.5*ATR(14) trailing
- Take profit: Reduce to half at 2R profit
- Position sizing: 0.25 base, discrete levels (0.0, ±0.25, ±0.30)

Why this should beat current best (Sharpe=0.490):
- Supertrend works well on 30m for crypto volatility
- RSI pullback zone (45-55) avoids chasing extremes
- Volume confirmation filters false breakouts
- Conservative sizing (0.25-0.30) controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_rsi_pullback_vol_30m_4h_v1"
timeframe = "30m"
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
    
    # Calculate HL2 and basic upper/lower bands
    hl2 = (high + low) / 2.0
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
            
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        # Final upper/lower calculation
        if i == period:
            final_upper[i] = upper_band[i]
            final_lower[i] = lower_band[i]
        else:
            if upper_band[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
                final_upper[i] = upper_band[i]
            else:
                final_upper[i] = final_upper[i - 1]
                
            if lower_band[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
                final_lower[i] = lower_band[i]
            else:
                final_lower[i] = final_lower[i - 1]
        
        # Determine trend
        if trend[i - 1] == 1:
            if close[i] < final_lower[i]:
                trend[i] = -1
                supertrend[i] = final_upper[i]
            else:
                trend[i] = 1
                supertrend[i] = final_lower[i]
        else:
            if close[i] > final_upper[i]:
                trend[i] = 1
                supertrend[i] = final_lower[i]
            else:
                trend[i] = -1
                supertrend[i] = final_upper[i]
    
    return supertrend, trend


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    delta = np.zeros(n)
    for i in range(1, n):
        delta[i] = close[i] - close[i - 1]
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rsi = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
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
    
    return adx


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    taker_buy_vol = prices["taker_buy_volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    
    # Calculate taker buy volume ratio
    taker_ratio = np.zeros(n)
    for i in range(n):
        if volume[i] > 0:
            taker_ratio[i] = taker_buy_vol[i] / volume[i]
        else:
            taker_ratio[i] = 0.5
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.30   # Max position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    entry_atr = 0.0
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(supertrend[i]) or
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]) or
            np.isnan(taker_ratio[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HTF trend filter
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        htf_trend = 1 if price_above_4h_hma else -1
        
        # Supertrend direction
        st_direction = st_trend[i]  # 1 = bullish, -1 = bearish
        
        # ADX regime filter (avoid chop)
        adx_strong = adx[i] > 20
        
        # RSI pullback zone (45-55 for entries in trend direction)
        rsi_pullback_long = 45 <= rsi[i] <= 55
        rsi_pullback_short = 45 <= rsi[i] <= 55
        
        # Volume confirmation
        vol_bullish = taker_ratio[i] > 0.55
        vol_bearish = taker_ratio[i] < 0.45
        
        # Determine target signal based on all filters
        target_signal = 0.0
        position_size = BASE_SIZE
        
        # Long entry: Supertrend bullish + HTF bullish + ADX strong + RSI pullback + Volume bullish
        if (st_direction == 1 and htf_trend == 1 and adx_strong and 
            rsi_pullback_long and vol_bullish):
            target_signal = position_size
        
        # Short entry: Supertrend bearish + HTF bearish + ADX strong + RSI pullback + Volume bearish
        elif (st_direction == -1 and htf_trend == -1 and adx_strong and 
              rsi_pullback_short and vol_bearish):
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2.5*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * entry_atr:  # 2R = 5*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 5.0 * entry_atr:  # 2R profit
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
                entry_atr = atr[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                # Exit if Supertrend reverses OR HTF alignment breaks
                st_reversal = (position_side == 1 and st_direction == -1) or \
                              (position_side == -1 and st_direction == 1)
                htf_alignment_broken = (position_side == 1 and htf_trend == -1) or \
                                       (position_side == -1 and htf_trend == 1)
                
                if st_reversal or htf_alignment_broken:
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
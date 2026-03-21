#!/usr/bin/env python3
"""
EXPERIMENT #013 - KAMA + Bollinger Regime + 1h Trend Filter (15m primary)
=====================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise better than 
EMA/HMA/Supertrend. Combined with Bollinger Band width percentile for regime detection 
(only trade when bands are expanding = trending regime), and 1h KAMA for trend filter, 
this should reduce false signals while capturing major moves. Volume confirmation 
(taker_buy_volume ratio) ensures we trade with institutional flow.

Key features:
- Primary TF: 15m
- HTF filter: 1h KAMA(21) for trend direction (faster than 4h for 15m entries)
- Trend: KAMA(21) adaptive moving average
- Regime: Bollinger Band width percentile > 60 (only trade in expanding volatility)
- Entry: KAMA crossover + volume confirmation + regime filter
- Strength: ADX(14) > 20 filter (lower threshold than Supertrend strategy)
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25 base, 0.35 max (discrete levels)
- Take profit: Reduce to half at 2R profit

Why this should beat previous attempts:
- KAMA adapts to volatility = fewer whipsaws in chop
- Bollinger regime filter = only trade when market is trending (not squeezing)
- 1h HTF = better alignment with 15m entries than 4h
- Volume confirmation = filter false breakouts
- Stricter entry = fewer trades, higher quality (target 500-2000 trades total)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_bbregime_1htrend_15m_v1"
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


def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise by adjusting smoothing constant based on efficiency ratio
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    change = np.zeros(n)
    volatility = np.zeros(n)
    
    for i in range(period, n):
        change[i] = abs(close[i] - close[i - period])
        vol_sum = 0.0
        for j in range(i - period + 1, i + 1):
            vol_sum += abs(close[j] - close[j - 1])
        volatility[i] = vol_sum
    
    er = np.zeros(n)
    for i in range(period, n):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = np.zeros(n)
    for i in range(period, n):
        sc[i] = er[i] * (fast_sc - slow_sc) + slow_sc
        sc[i] = sc[i] ** 2  # Square for smoother adaptation
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and bandwidth"""
    n = len(close)
    close_s = pd.Series(close)
    
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    
    upper_band = sma + std_dev * std
    lower_band = sma - std_dev * std
    bandwidth = (upper_band - lower_band) / sma
    
    return sma, upper_band, lower_band, bandwidth


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


def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)"""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def calculate_bw_percentile(bandwidth, lookback=100):
    """Calculate Bollinger Band width percentile rank"""
    n = len(bandwidth)
    bw_pct = np.zeros(n)
    bw_pct[:] = np.nan
    
    for i in range(lookback, n):
        window = bandwidth[i - lookback:i + 1]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) > 0:
            rank = np.sum(valid_window < bandwidth[i])
            bw_pct[i] = rank / len(valid_window) * 100
    
    return bw_pct


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    taker_buy_vol = prices["taker_buy_volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 1h KAMA for trend filter
    kama_1h = calculate_kama(df_1h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    kama_1h_aligned = align_htf_to_ltf(prices, df_1h, kama_1h)
    
    # Calculate 15m indicators
    kama_15m = calculate_kama(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    # Bollinger Bands for regime detection
    bb_sma, bb_upper, bb_lower, bb_bandwidth = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    bw_percentile = calculate_bw_percentile(bb_bandwidth, lookback=100)
    
    # Volume ratio (taker buy / total volume)
    volume_ratio = np.zeros(n)
    for i in range(n):
        if volume[i] > 0:
            volume_ratio[i] = taker_buy_vol[i] / volume[i]
        else:
            volume_ratio[i] = 0.5
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.35   # Max position size with strong signals
    MIN_SIZE = 0.20   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(kama_1h_aligned[i]) or np.isnan(kama_15m[i]) or
            np.isnan(atr[i]) or np.isnan(adx[i]) or np.isnan(bw_percentile[i]) or
            np.isnan(volume_ratio[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 1h KAMA trend filter
        hma_trend = 1 if close[i] > kama_1h_aligned[i] else -1
        
        # 15m KAMA trend
        kama_trend_15m = 1 if close[i] > kama_15m[i] else -1
        
        # KAMA crossover signal
        kama_crossover_long = (close[i] > kama_15m[i]) and (close[i - 1] <= kama_15m[i - 1])
        kama_crossover_short = (close[i] < kama_15m[i]) and (close[i - 1] >= kama_15m[i - 1])
        
        # Bollinger Band regime filter (only trade when bandwidth is expanding)
        regime_expanding = bw_percentile[i] > 60  # Top 40% of bandwidth = trending
        
        # ADX strength filter (lower threshold than Supertrend strategy)
        adx_strong = adx[i] > 20
        
        # Volume confirmation (institutional flow)
        volume_bullish = volume_ratio[i] > 0.55  # More buying pressure
        volume_bearish = volume_ratio[i] < 0.45  # More selling pressure
        
        # RSI filter (avoid overbought/oversold entries)
        rsi_ok_long = rsi[i] < 70  # Not overbought
        rsi_ok_short = rsi[i] > 30  # Not oversold
        
        # DI+ vs DI- for trend confirmation
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = minus_di[i] > plus_di[i]
        
        # Calculate position size based on ADX strength (dynamic sizing)
        adx_multiplier = min(1.0 + (adx[i] - 20) / 50, 1.4)  # Max 1.4x
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * adx_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: KAMA crossover + 1h trend bullish + regime expanding + ADX strong + volume bullish
        if (kama_crossover_long and hma_trend == 1 and kama_trend_15m == 1 and
            regime_expanding and adx_strong and volume_bullish and rsi_ok_long and di_bullish):
            target_signal = position_size
        
        # Short entry: KAMA crossover + 1h trend bearish + regime expanding + ADX strong + volume bearish
        elif (kama_crossover_short and hma_trend == -1 and kama_trend_15m == -1 and
              regime_expanding and adx_strong and volume_bearish and rsi_ok_short and di_bearish):
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                
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
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                
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
                entry_atr = atr[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                # Exit if KAMA trend reverses OR 1h KAMA alignment breaks
                kama_reversal_long = kama_trend_15m == -1
                kama_reversal_short = kama_trend_15m == 1
                hma_alignment_broken = (position_side == 1 and hma_trend == -1) or \
                                       (position_side == -1 and hma_trend == 1)
                
                if kama_reversal_long or kama_reversal_short or hma_alignment_broken:
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
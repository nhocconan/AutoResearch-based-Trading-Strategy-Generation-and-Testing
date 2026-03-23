#!/usr/bin/env python3
"""
Experiment #720: 1h Primary + 4h/12h HTF — Fisher Transform + KAMA Trend

Hypothesis: 1h timeframe with HTF trend bias can work IF:
1. Entry filters are LOOSE enough to generate 30-60 trades/year
2. Fisher Transform catches reversals better than RSI in bear markets
3. KAMA adapts to volatility better than HMA (less whipsaw)
4. 4h trend direction prevents counter-trend disasters

Why this should beat current best (Sharpe=0.612):
- Fisher Transform is proven for reversal capture in ranging/bear markets
- KAMA reduces false signals during low-volatility periods
- 1h entries within 4h trend = proven pattern from best strategies
- Simpler logic = more trades (avoid 0-trade failures #710, #715)

Key differences from failed 1h strategies:
- NO session filter (killed trade frequency in #710, #715)
- NO Choppiness Index (too many conditions = 0 trades)
- NO CRSI (failed in #708, #710, #712)
- Fisher + KAMA = new combination not yet tried

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_kama_trend_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market noise."""
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Efficiency Ratio
    signal = np.abs(close - np.roll(close, er_period))
    signal[:er_period] = np.nan
    
    noise = np.zeros(n)
    for i in range(er_period, n):
        noise[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        er = signal / (noise + 1e-10)
    er = np.clip(er, 0, 1)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, close, period=9):
    """Ehlers Fisher Transform - normalizes price for reversal detection."""
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period * 2:
        return fisher, fisher_signal
    
    # Calculate typical price and normalize
    hl2 = (high + low) / 2
    highest = pd.Series(hl2).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(hl2).rolling(window=period, min_periods=period).min().values
    
    # Normalize to -1 to +1 range
    with np.errstate(divide='ignore', invalid='ignore'):
        norm = 2 * (hl2 - lowest) / (highest - lowest + 1e-10) - 1
    norm = np.clip(norm, -0.999, 0.999)
    
    # Fisher transform
    fisher_raw = 0.5 * np.log((1 + norm) / (1 - norm + 1e-10))
    fisher = pd.Series(fisher_raw).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    # Signal line (previous fisher)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = np.nan
    
    return fisher, fisher_signal

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs moving average."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = volume / (vol_sma + 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate primary (1h) indicators
    kama_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, period=200)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Calculate and align HTF KAMA for trend bias
    kama_4h_raw = calculate_kama(df_4h['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h_raw)
    
    kama_12h_raw = calculate_kama(df_12h['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Need 200 for SMA + buffer for HTF alignment
        # Skip if indicators not ready
        if np.isnan(kama_1h[i]) or np.isnan(fisher[i]) or np.isnan(atr_1h[i]):
            continue
        if np.isnan(rsi_1h[i]) or np.isnan(sma_200[i]) or np.isnan(vol_ratio[i]):
            continue
        if np.isnan(kama_4h_aligned[i]) or np.isnan(kama_12h_aligned[i]):
            continue
        if atr_1h[i] <= 1e-10:
            continue
        
        # === HTF TREND BIAS (4h + 12h KAMA) ===
        # Both HTFs must agree for strong signal
        htf_bullish = (close[i] > kama_4h_aligned[i]) and (close[i] > kama_12h_aligned[i])
        htf_bearish = (close[i] < kama_4h_aligned[i]) and (close[i] < kama_12h_aligned[i])
        htf_neutral = not htf_bullish and not htf_bearish
        
        # === LTF TREND (1h KAMA) ===
        ltf_bullish = close[i] > kama_1h[i]
        ltf_bearish = close[i] < kama_1h[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.5 from below = long reversal signal
        # Fisher crosses below +1.5 from above = short reversal signal
        fisher_long = (fisher[i] < -1.0) and (fisher_signal[i] < fisher[i])
        fisher_short = (fisher[i] > 1.0) and (fisher_signal[i] > fisher[i])
        
        # Extreme Fisher values = strong reversal potential
        fisher_extreme_long = fisher[i] < -1.5
        fisher_extreme_short = fisher[i] > 1.5
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_1h[i] < 40
        rsi_overbought = rsi_1h[i] > 60
        rsi_extreme_oversold = rsi_1h[i] < 30
        rsi_extreme_overbought = rsi_1h[i] > 70
        
        # === VOLUME CONFIRMATION ===
        volume_ok = vol_ratio[i] > 0.7  # At least 70% of average volume
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === LONG ENTRY CONDITIONS ===
        # Primary: HTF bullish + Fisher long signal + RSI confirmation
        if htf_bullish and fisher_long and rsi_oversold and volume_ok:
            desired_signal = current_size
        
        # Secondary: HTF bullish + Fisher extreme + above SMA200
        elif htf_bullish and fisher_extreme_long and above_sma200:
            desired_signal = current_size
        
        # Tertiary: LTF bullish + Fisher long + RSI extreme (counter-HTF but strong)
        elif ltf_bullish and fisher_extreme_long and rsi_extreme_oversold:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY CONDITIONS ===
        # Primary: HTF bearish + Fisher short signal + RSI confirmation
        if htf_bearish and fisher_short and rsi_overbought and volume_ok:
            desired_signal = -current_size
        
        # Secondary: HTF bearish + Fisher extreme + below SMA200
        elif htf_bearish and fisher_extreme_short and below_sma200:
            desired_signal = -current_size
        
        # Tertiary: LTF bearish + Fisher extreme + RSI extreme (counter-HTF but strong)
        elif ltf_bearish and fisher_extreme_short and rsi_extreme_overbought:
            desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if Fisher not overbought and HTF trend intact
                if fisher[i] < 1.5 and htf_bullish:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if Fisher not oversold and HTF trend intact
                if fisher[i] > -1.5 and htf_bearish:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if Fisher overbought or HTF trend reverses
            if fisher[i] > 1.8:
                desired_signal = 0.0
            elif close[i] < kama_4h_aligned[i] and close[i] < kama_12h_aligned[i]:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if Fisher oversold or HTF trend reverses
            if fisher[i] < -1.8:
                desired_signal = 0.0
            elif close[i] > kama_4h_aligned[i] and close[i] > kama_12h_aligned[i]:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE * 0.8 else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE * 0.8 else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals
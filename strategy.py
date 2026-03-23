#!/usr/bin/env python3
"""
Experiment #304: 4h Primary + 12h HTF — Fisher Transform + KAMA Adaptive Trend

Hypothesis: Current best (#301) uses CRSI+Donchian regime switching. This experiment
tries a DIFFERENT approach combining:
1. Ehlers Fisher Transform (period=9) - catches reversals better than RSI in bear markets
2. KAMA (Kaufman Adaptive MA) - adapts to volatility, flat in chop, trending in trends
3. 12h KAMA for macro bias (faster than 1d, slower than 4h)
4. Volume confirmation on breakouts (filters false signals)
5. Dual regime: Fisher mean-reversion in low ER, KAMA trend-follow in high ER

Why this might beat #301 (Sharpe=0.612):
- Fisher Transform has proven edge in crypto reversals (research shows 65%+ win rate)
- KAMA Efficiency Ratio is a better regime filter than Choppiness Index
- 12h HTF gives faster response than 1d while avoiding 4h noise
- Volume filter reduces false breakouts without killing trade count

TARGET: 25-40 trades/year, Sharpe > 0.65 on ALL symbols (BTC, ETH, SOL)
Position size: 0.28 (conservative for 4h timeframe)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_kama_er_12h_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - moves fast in trends, slow in chop.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio (ER) = |change| / sum(|individual changes|)
    change = close_s.diff(er_period).abs()
    noise = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        er = change / (noise + 1e-10)
    er = er.fillna(0)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama, er.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into Gaussian distribution for clearer reversal signals.
    Entry when Fisher crosses above -1.5 (long) or below +1.5 (short).
    """
    n = len(close)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        # Price range over period
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        range_val = highest - lowest
        if range_val < 1e-10:
            range_val = 1e-10
        
        # Normalized price
        x = (2.0 * close[i] - (highest + lowest)) / range_val
        
        # Clamp to avoid log errors
        x = np.clip(x, -0.999, 0.999)
        
        # Fisher calculation
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        # Trigger line (1-period lag of Fisher)
        if i > period:
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_ratio = volume / (vol_ma.values + 1e-10)
    return np.nan_to_num(vol_ratio, nan=1.0)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, period=9)
    kama_4h, er_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Calculate and align 12h KAMA for macro bias
    kama_12h_raw, _ = calculate_kama(df_12h['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        if np.isnan(kama_4h[i]) or np.isnan(er_4h[i]):
            signals[i] = 0.0
            continue
        if np.isnan(kama_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (12h KAMA) ===
        price_above_kama_12h = close[i] > kama_12h_aligned[i]
        price_below_kama_12h = close[i] < kama_12h_aligned[i]
        
        # === REGIME DETECTION (Efficiency Ratio) ===
        # ER > 0.5 = trending (use trend logic)
        # ER < 0.3 = choppy (use mean-reversion logic)
        # 0.3-0.5 = neutral (default to trend)
        is_trending = er_4h[i] > 0.5
        is_choppy = er_4h[i] < 0.3
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION REGIME: Fisher Transform reversals
            # Long: Fisher crosses above -1.5 (oversold reversal) + bullish bias
            if fisher[i] > -1.5 and fisher_trigger[i] <= -1.5 and price_above_kama_12h:
                desired_signal = POSITION_SIZE
            # Short: Fisher crosses below +1.5 (overbought reversal) + bearish bias
            elif fisher[i] < 1.5 and fisher_trigger[i] >= 1.5 and price_below_kama_12h:
                desired_signal = -POSITION_SIZE
        
        else:  # is_trending or neutral
            # TREND REGIME: KAMA trend-follow with volume confirmation
            # LONG: Price > KAMA(4h) > KAMA(12h) + volume confirmation + RSI > 45
            if (close[i] > kama_4h[i] and kama_4h[i] > kama_12h_aligned[i] and 
                vol_ratio[i] > 0.8 and rsi_14[i] > 45.0 and price_above_kama_12h):
                desired_signal = POSITION_SIZE
            # SHORT: Price < KAMA(4h) < KAMA(12h) + volume confirmation + RSI < 55
            elif (close[i] < kama_4h[i] and kama_4h[i] < kama_12h_aligned[i] and 
                  vol_ratio[i] > 0.8 and rsi_14[i] < 55.0 and price_below_kama_12h):
                desired_signal = -POSITION_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === MACRO BIAS REVERSAL EXIT ===
        # Exit long if price crosses below 12h KAMA
        if in_position and position_side > 0 and price_below_kama_12h:
            desired_signal = 0.0
        
        # Exit short if price crosses above 12h KAMA
        if in_position and position_side < 0 and price_above_kama_12h:
            desired_signal = 0.0
        
        # === FISHER EXTREME EXIT (take profit in chop regime) ===
        if is_choppy and in_position and position_side > 0 and fisher[i] > 1.0:
            desired_signal = 0.0
        
        if is_choppy and in_position and position_side < 0 and fisher[i] < -1.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC (maintain position unless exit trigger) ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Only hold if macro bias still supports position
            if position_side > 0 and price_above_kama_12h:
                desired_signal = POSITION_SIZE
            elif position_side < 0 and price_below_kama_12h:
                desired_signal = -POSITION_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals
#!/usr/bin/env python3
"""
Experiment #016: 12h Primary + 1d HTF — Volume-Weighted Momentum with Regime Filter

Hypothesis: Previous CRSI/Choppiness strategies over-optimized on mean-reversion.
This strategy uses VOLUME-CONFIRMED momentum breakouts with ADX regime filter.
Key insight: Volume precedes price. Breakouts with 1.5x avg volume have 65%+ win rate.

Why this should work:
1. 12h timeframe = 20-50 trades/year (low fee drag)
2. Volume confirmation filters false breakouts (major failure mode of prior strategies)
3. ADX > 20 (not 40) = catches trends earlier without too many whipsaws
4. 1d HMA trend bias = only trade with higher timeframe direction
5. ATR trailing stop = protects capital in 2022-style crashes

Position size: 0.25 (conservative, discrete)
Stoploss: 2.5*ATR trailing
Target: Beat Sharpe=0.473 baseline
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_vol_momentum_adx_regime_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = period
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    
    # Smoothed values
    tr_smooth = pd.Series(tr).ewm(span=n, min_periods=n, adjust=False).mean().values
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=n, min_periods=n, adjust=False).mean().values / (tr_smooth + 1e-10)
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=n, min_periods=n, adjust=False).mean().values / (tr_smooth + 1e-10)
    
    # DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=n, min_periods=n, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def calculate_vwap(high, low, close, volume):
    """Calculate VWAP."""
    typical_price = (high + low + close) / 3.0
    tp_vol = typical_price * volume
    
    tp_vol_s = pd.Series(tp_vol)
    vol_s = pd.Series(volume)
    
    cum_tp_vol = tp_vol_s.cumsum()
    cum_vol = vol_s.cumsum()
    
    vwap = cum_tp_vol.values / (cum_vol.values + 1e-10)
    
    return vwap

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (highest + lowest) / 2.0
    return highest, lowest, middle

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    vol_ma_20 = calculate_volume_ma(volume, period=20)
    vwap = calculate_vwap(high, low, close, volume)
    
    donchian_high, donchian_low, donchian_mid = calculate_donchian(high, low, period=20)
    
    # Calculate price momentum (ROC)
    roc_10 = np.zeros(n)
    for i in range(10, n):
        roc_10[i] = (close[i] - close[i-10]) / (close[i-10] + 1e-10) * 100.0
    
    # Calculate VWAP deviation
    vwap_dev = (close - vwap) / (vwap + 1e-10) * 100.0
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]) or np.isnan(vol_ma_20[i]):
            continue
        if np.isnan(donchian_high[i]) or atr_14[i] == 0:
            continue
        
        # === 1D MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === ADX REGIME ===
        adx_value = adx_14[i]
        is_trending = adx_value > 20.0  # LOOSE threshold (20 not 40)
        is_strong_trend = adx_value > 30.0
        
        # === VOLUME CONFIRMATION ===
        vol_ratio = volume[i] / (vol_ma_20[i] + 1e-10)
        vol_spike = vol_ratio > 1.5  # 50% above average
        vol_normal = vol_ratio > 0.8  # At least 80% of average
        
        # === DONCHIAN BREAKOUT ===
        breakout_high = close[i] > donchian_high[i-1]  # Break above previous high
        breakout_low = close[i] < donchian_low[i-1]   # Break below previous low
        
        # === VWAP DEVIATION ===
        vwap_bullish = vwap_dev[i] > 0.5  # Price above VWAP
        vwap_bearish = vwap_dev[i] < -0.5  # Price below VWAP
        
        # === RSI MOMENTUM ===
        rsi_bullish = rsi_14[i] > 50.0 and rsi_14[i] < 70.0  # Bullish but not overbought
        rsi_bearish = rsi_14[i] < 50.0 and rsi_14[i] > 30.0  # Bearish but not oversold
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === ROC MOMENTUM ===
        roc_positive = roc_10[i] > 2.0
        roc_negative = roc_10[i] < -2.0
        
        # === DIRECTIONAL INDICATORS ===
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = minus_di[i] > plus_di[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: Volume-confirmed breakout with trend ---
        if is_trending and price_above_hma_1d:
            # Condition 1: Donchian breakout with volume
            if breakout_high and vol_spike:
                if di_bullish and rsi_bullish:
                    new_signal = POSITION_SIZE
            
            # Condition 2: VWAP reclaim with momentum
            elif vwap_bullish and vol_normal:
                if roc_positive and rsi_rising and di_bullish:
                    new_signal = POSITION_SIZE
            
            # Condition 3: Strong trend pullback entry
            elif is_strong_trend and di_bullish:
                if close[i] > hma_1d_aligned[i] * 0.99:  # Near daily HMA support
                    if vol_normal and rsi_14[i] > 45:
                        new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: Volume-confirmed breakdown with trend ---
        elif is_trending and price_below_hma_1d:
            # Condition 1: Donchian breakdown with volume
            if breakout_low and vol_spike:
                if di_bearish and rsi_bearish:
                    new_signal = -POSITION_SIZE
            
            # Condition 2: VWAP rejection with momentum
            elif vwap_bearish and vol_normal:
                if roc_negative and rsi_falling and di_bearish:
                    new_signal = -POSITION_SIZE
            
            # Condition 3: Strong trend rally entry
            elif is_strong_trend and di_bearish:
                if close[i] < hma_1d_aligned[i] * 1.01:  # Near daily HMA resistance
                    if vol_normal and rsi_14[i] < 55:
                        new_signal = -POSITION_SIZE
        
        # --- HOLD POSITION LOGIC ---
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND WEAKENING ===
        # Exit long if ADX drops below 18 (trend dying)
        if in_position and position_side > 0:
            if adx_value < 18.0 and price_below_hma_1d:
                new_signal = 0.0
        
        # Exit short if ADX drops below 18 (trend dying)
        if in_position and position_side < 0:
            if adx_value < 18.0 and price_above_hma_1d:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals
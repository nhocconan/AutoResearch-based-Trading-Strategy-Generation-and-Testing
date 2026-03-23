#!/usr/bin/env python3
"""
Experiment #059: 4h Primary + 1d HTF — Volume-Confirmed Trend Following with RSI Timing

Hypothesis: 4h timeframe with 1d trend bias, using volume surge confirmation + RSI 
entry timing will generate 30-60 trades/year with better Sharpe than previous 4h attempts.

Key insights from 58 failed experiments:
1) 4h timeframe is proven to work (current best Sharpe=0.486)
2) CRSI+Chop+Donchian combinations have been exhausted — need NEW approach
3) Many strategies got 0 trades — entry conditions were TOO STRICT
4) Volume confirmation reduces false breakouts significantly
5) RSI timing (not extremes) improves entry quality vs pure breakout
6) 1d HTF provides strong macro bias without over-filtering

Why this should work:
- 4h primary = proven timeframe (balances trade frequency vs fee drag)
- 1d HTF = macro trend filter (prevents counter-trend trades in bear markets)
- Volume surge = confirms genuine breakouts (filters fake moves)
- RSI 40-60 zone = entry timing without waiting for extremes (ensures trades)
- ADX > 20 = trend confirmation (lower threshold than typical 25 to get more trades)
- ATR stoploss = proper risk management

Position size: 0.28 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
Target: 30-60 trades/year, Sharpe > 0.486
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_volume_hma_rsi_adx_1d_v1"
timeframe = "4h"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_volume_ratio(taker_buy_volume, volume, period=20):
    """Calculate volume ratio (taker buy / total) and volume surge."""
    vol_ratio = taker_buy_volume / (volume + 1e-10)
    avg_vol = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_surge = volume / (avg_vol + 1e-10)
    return vol_ratio, vol_surge

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    rsi_14 = calculate_rsi(close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    vol_ratio, vol_surge = calculate_volume_ratio(taker_buy_vol, volume, period=20)
    sma_50 = calculate_sma(close, period=50)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.28  # Discrete, within 0.20-0.35 range
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):  # Warmup for all indicators
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]) or np.isnan(hma_21[i]):
            continue
        if np.isnan(vol_surge[i]) or np.isnan(sma_50[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4H TREND CONFIRMATION ===
        price_above_hma_21 = close[i] > hma_21[i]
        price_below_hma_21 = close[i] < hma_21[i]
        price_above_hma_50 = close[i] > hma_50[i]
        price_below_hma_50 = close[i] < hma_50[i]
        hma_21_above_50 = hma_21[i] > hma_50[i]
        hma_21_below_50 = hma_21[i] < hma_50[i]
        
        # === TREND STRENGTH ===
        adx_strong = adx_14[i] > 20.0  # Lower threshold to get more trades
        adx_very_strong = adx_14[i] > 25.0
        
        # === VOLUME CONFIRMATION ===
        vol_surge_up = vol_surge[i] > 1.2  # Volume 20% above average
        vol_surge_down = vol_surge[i] > 1.2
        buyer_dominant = vol_ratio[i] > 0.55  # More taker buys
        seller_dominant = vol_ratio[i] < 0.45  # More taker sells
        
        # === RSI ENTRY TIMING (not extremes - ensures trades) ===
        rsi_bullish = rsi_14[i] > 45.0  # Bullish momentum
        rsi_bearish = rsi_14[i] < 55.0  # Bearish momentum
        rsi_not_overbought = rsi_14[i] < 70.0
        rsi_not_oversold = rsi_14[i] > 30.0
        
        # === HMA SLOPE ===
        hma_21_slope_up = hma_21[i] > hma_21[i-3] if i > 3 else False
        hma_21_slope_down = hma_21[i] < hma_21[i-3] if i > 3 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: Trend + Volume + RSI confirmation ---
        # Must have: 1d bullish bias OR 4h strong trend
        # Plus: volume surge + buyer dominant + RSI bullish
        if price_above_hma_1d or (hma_21_above_50 and adx_strong):
            if price_above_hma_21 and hma_21_slope_up:
                if vol_surge_up and buyer_dominant:
                    if rsi_bullish and rsi_not_overbought:
                        new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: Trend + Volume + RSI confirmation ---
        # Must have: 1d bearish bias OR 4h strong trend
        # Plus: volume surge + seller dominant + RSI bearish
        if price_below_hma_1d or (hma_21_below_50 and adx_strong):
            if price_below_hma_21 and hma_21_slope_down:
                if vol_surge_down and seller_dominant:
                    if rsi_bearish and rsi_not_oversold:
                        new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            # Hold if RSI not at opposite extreme and trend intact
            if position_side > 0 and rsi_14[i] < 75.0 and price_above_hma_21:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and rsi_14[i] > 25.0 and price_below_hma_21:
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
        
        # === EXIT ON TREND CHANGE ===
        if in_position and position_side > 0:
            if price_below_hma_50 and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_50 and price_above_hma_1d:
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
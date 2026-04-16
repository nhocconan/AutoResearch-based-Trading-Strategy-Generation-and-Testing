#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d RSI(14) mean reversion with 4h volume confirmation and ATR trailing stop.
# Long when 1d RSI < 30 (oversold) and 4h close > 4h open (bullish candle) with volume > 1.5x median volume.
# Short when 1d RSI > 70 (overbought) and 4h close < 4h open (bearish candle) with volume > 1.5x median volume.
# Uses discrete position size 0.25. Exits when 1d RSI crosses 50 (mean reversion) or ATR stoploss hits (2.0x ATR).
# 1d RSI identifies extremes; 4h price action and volume filter ensure momentum alignment.
# Targets 20-50 trades/year to minimize fee drag while capturing reversals in both bull/bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: RSI (14-period) ===
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # === 4h Indicators: Volume Median (20-period) ===
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # === 4h Indicators: ATR (14-period) for stoploss ===
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to primary timeframe (4h)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(14, 20)  # RSI, Volume median
    
    # Track position state and entry price for ATR stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_aligned[i]) or np.isnan(vol_median_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values (aligned)
        price = close[i]
        open_price = open_[i]
        rsi_val = rsi_aligned[i]
        vol_median = vol_median_aligned[i]
        atr = atr_14[i]
        
        # Volume spike filter: current 4h volume > 1.5x median volume
        volume_spike = volume[i] > (vol_median * 1.5)
        
        # Bullish/bearish candle
        bullish_candle = price > open_price
        bearish_candle = price < open_price
        
        # RSI thresholds
        rsi_oversold = rsi_val < 30
        rsi_overbought = rsi_val > 70
        rsi_exit = (rsi_val > 50 and position == 1) or (rsi_val < 50 and position == -1)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when RSI crosses above 50 (mean reversion) OR ATR stoploss hit (2.0 * ATR below entry)
            if rsi_exit or price <= entry_price - 2.0 * atr:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when RSI crosses below 50 (mean reversion) OR ATR stoploss hit (2.0 * ATR above entry)
            if rsi_exit or price >= entry_price + 2.0 * atr:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: RSI oversold + bullish candle + volume spike
            if rsi_oversold and bullish_candle and volume_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: RSI overbought + bearish candle + volume spike
            elif rsi_overbought and bearish_candle and volume_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_1dRSI14_OBOS_4hBullBearCandle_VolumeSpike1.5x_EXIT50_ATRTrail2.0_v1"
timeframe = "4h"
leverage = 1.0
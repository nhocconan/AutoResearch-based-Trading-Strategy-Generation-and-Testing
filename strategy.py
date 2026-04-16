#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d ATR volatility filter and volume confirmation.
# Long when price breaks above Camarilla R4 AND 1d ATR(14) > 1.2x 50-period MA AND volume > 1.5x 20-period average.
# Short when price breaks below Camarilla S4 AND 1d ATR(14) > 1.2x 50-period MA AND volume > 1.5x 20-period average.
# Exit on opposite Camarilla level (R3 for longs, S3 for shorts) or ATR-based stop (1.5*ATR from entry).
# Uses discrete position size 0.25. Designed to capture strong volatility expansions in both bull and bear markets.
# Camarilla levels provide adaptive support/resistance based on prior day's range, effective in ranging and trending markets.
# Volatility filter ensures trades only during elevated volatility regimes, reducing false breakouts.
# Volume confirmation adds conviction to breakouts.
# Target: 75-150 total trades over 4 years (19-38/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Camarilla Pivot Levels (based on prior 6h bar) ===
    # Camarilla levels calculated from prior bar's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    
    camarilla_r3 = pivot + (range_ * 1.1 / 4.0)
    camarilla_s3 = pivot - (range_ * 1.1 / 4.0)
    camarilla_r4 = pivot + (range_ * 1.1 / 2.0)
    camarilla_s4 = pivot - (range_ * 1.1 / 2.0)
    
    # === 1d Indicators: ATR(14) and its 50-period MA for volatility regime ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d_raw = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # 50-period MA of ATR
    atr_ma_1d = pd.Series(atr_1d_raw).rolling(window=50, min_periods=50).mean().values
    atr_1d = align_htf_to_ltf(prices, df_1d, atr_1d_raw)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    high_volatility = atr_1d > (1.2 * atr_ma_1d_aligned)
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 6h ATR for stoploss ===
    tr1_6h = pd.Series(high).diff()
    tr2_6h = pd.Series(low).diff().abs()
    tr3_6h = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1_6h, tr2_6h, tr3_6h], axis=1).max(axis=1)
    atr_6h_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 60 periods needed for ATR MA/volume MA)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i]) or np.isnan(high_volatility[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_6h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        high_vol = high_volatility[i]
        vol_spike = volume_spike[i]
        atr_val = atr_6h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Camarilla S3 (profit target/reversal)
            if price < camarilla_s3[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR below entry
            elif price < entry_price - 1.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla R3 (profit target/reversal)
            if price > camarilla_r3[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR above entry
            elif price > entry_price + 1.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R4 AND high volatility AND volume spike
            if price > camarilla_r4[i] and high_vol and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S4 AND high volatility AND volume spike
            elif price < camarilla_s4[i] and high_vol and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Camarilla_R4_S4_1dATRVol_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0
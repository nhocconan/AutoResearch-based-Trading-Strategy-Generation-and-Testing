#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h chart with 4h trend filter (EMA21) and 1d volatility filter (ATR-based).
# Long when price > 4h EMA21 and price closes above 1h Bollinger Upper Band with volume spike.
# Short when price < 4h EMA21 and price closes below 1h Bollinger Lower Band with volume spike.
# Uses 1d ATR to filter out low volatility periods and avoid whipsaws.
# Target: 15-35 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for EMA21 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Load 1d data for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.abs(high_1d[0] - low_1d[0])
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 1h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    # Volume filter: current volume > 2.0 x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    # Volatility filter: avoid low volatility periods
    vol_filter = atr_14_1d_aligned > (0.5 * pd.Series(atr_14_1d_aligned).rolling(window=50, min_periods=50).mean().values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(vol_spike[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend = ema_21_4h_aligned[i]
        atr_vol = atr_14_1d_aligned[i]
        upper = upper_band[i]
        lower = lower_band[i]
        vol_ok = vol_spike[i]
        vol_regime = vol_filter[i]
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long: price > 4h EMA21, price closes above upper BB, volume spike, normal volatility
            if price > ema_trend and close[i] > upper and vol_ok and vol_regime and in_session:
                signals[i] = 0.20
                position = 1
            # Short: price < 4h EMA21, price closes below lower BB, volume spike, normal volatility
            elif price < ema_trend and close[i] < lower and vol_ok and vol_regime and in_session:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price closes below 4h EMA21 or volatility too high
            if close[i] < ema_trend or not vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price closes above 4h EMA21 or volatility too high
            if close[i] > ema_trend or not vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_EMA21_1d_ATR_Volume_BB_Breakout_VolumeFilter_v1"
timeframe = "1h"
leverage = 1.0
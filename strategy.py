#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Force Index with 1d RSI filter and volume confirmation.
# Long when EFI > 0 and RSI < 40 with volume > 1.3x average.
# Short when EFI < 0 and RSI > 60 with volume > 1.3x average.
# EFI combines price movement and volume for momentum confirmation.
# RSI filter prevents overextended entries. Volume ensures conviction.
# Target: 60-120 total trades over 4 years (15-30/year) for optimal frequency.

name = "6h_elders_force_1d_rsi_vol_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Force Index (13-period EMA of price change * volume)
    price_change = np.diff(close, prepend=close[0])
    raw_efi = price_change * volume
    efi_series = pd.Series(raw_efi)
    efi = efi_series.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Daily RSI (14-period)
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    delta = np.diff(daily_close, prepend=daily_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    avg_gain = gain_series.ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if RSI data not available
        if np.isnan(rsi_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits
        if position == 1:  # long position
            # Exit: EFI turns negative or RSI > 70 (overbought)
            if (efi[i] <= 0 or 
                rsi_aligned[i] >= 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: EFI turns positive or RSI < 30 (oversold)
            if (efi[i] >= 0 or 
                rsi_aligned[i] <= 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Long: positive EFI and RSI < 40 (not overbought)
                if (efi[i] > 0 and 
                    rsi_aligned[i] < 40):
                    signals[i] = 0.25
                    position = 1
                # Short: negative EFI and RSI > 60 (not oversold)
                elif (efi[i] < 0 and 
                      rsi_aligned[i] > 60):
                    signals[i] = -0.25
                    position = -1
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and ATR-scaled volume confirmation
# Long when price breaks above Camarilla R3 AND 1d close > 1d EMA34 (uptrend) AND volume > 2.0 * 20-bar ATR-scaled volume
# Short when price breaks below Camarilla S3 AND 1d close < 1d EMA34 (downtrend) AND volume > 2.0 * 20-bar ATR-scaled volume
# Exit when price retraces to the Camarilla H5/L5 level (mean reversion zone)
# Uses discrete sizing 0.30 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Camarilla levels provide intraday support/resistance; 1d EMA34 filters trend; volume confirmation avoids false breakouts

name = "12h_Camarilla_R3S3_1dEMA34_ATRVolume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for 12h timeframe (based on previous bar)
    # Camarilla: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # R3 = H4 + 1.1*(H-L)/2, S3 = L4 - 1.1*(H-L)/2
    # H5 = C + 1.1*(H-L)/2, L5 = C - 1.1*(H-L)/2 (same as H4/L4)
    # We use previous bar to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    rang = prev_high - prev_low
    camarilla_h4 = prev_close + 1.1 * rang / 2.0
    camarilla_l4 = prev_close - 1.1 * rang / 2.0
    camarilla_r3 = camarilla_h4 + 1.1 * rang / 2.0  # R3 = H4 + 1.1*(H-L)/2
    camarilla_s3 = camarilla_l4 - 1.1 * rang / 2.0  # S3 = L4 - 1.1*(H-L)/2
    camarilla_h5 = camarilla_h4  # H5 = H4
    camarilla_l5 = camarilla_l4  # L5 = L4
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe (wait for completed HTF bar)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate ATR for volume confirmation (14-period)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = np.nan
    tr3.iloc[0] = np.nan
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR-scaled volume: volume > 2.0 * 20-bar average of (volume / ATR)
    # Avoid division by zero or near-zero ATR
    atr_safe = np.where(atr < 1e-10, np.nan, atr)
    volume_per_atr = volume / atr_safe
    avg_volume_per_atr_20 = pd.Series(volume_per_atr).rolling(window=20, min_periods=20).mean().values
    volume_confirmation = volume_per_atr > (2.0 * avg_volume_per_atr_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(camarilla_h5[i]) or 
            np.isnan(camarilla_l5[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_confirmation[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Camarilla breakout signals with trend and volume filters
            # Long: Break above R3 AND uptrend AND volume confirmation
            if close[i] > camarilla_r3[i] and close[i] > ema34_1d_aligned[i] and volume_confirmation[i]:
                signals[i] = 0.30
                position = 1
            # Short: Break below S3 AND downtrend AND volume confirmation
            elif close[i] < camarilla_s3[i] and close[i] < ema34_1d_aligned[i] and volume_confirmation[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Price retraces to H5/L5 level (mean reversion zone)
            if close[i] <= camarilla_h5[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Price retraces to H5/L5 level (mean reversion zone)
            if close[i] >= camarilla_l5[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_volatility_breakout_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate daily volatility range (ATR-based)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily ATR (14-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily volatility bands (mean ± 1.5 * ATR)
    ma_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    upper_1d = ma_1d + 1.5 * atr_1d
    lower_1d = ma_1d - 1.5 * atr_1d
    
    # Align daily volatility bands to 12h timeframe
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    ma_1d_aligned = align_htf_to_ltf(prices, df_1d, ma_1d)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: price above/below 50-period EMA on 12h
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or
            np.isnan(ma_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_50[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        upper = upper_1d_aligned[i]
        lower = lower_1d_aligned[i]
        ma = ma_1d_aligned[i]
        ema = ema_50[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.3 * vol_ma
        
        # Entry signals
        long_signal = False
        short_signal = False
        
        # Long: price breaks above upper volatility band with volume and above EMA50
        if price_high > upper and volume_confirmed and price_close > ema:
            long_signal = True
        
        # Short: price breaks below lower volatility band with volume and below EMA50
        if price_low < lower and volume_confirmed and price_close < ema:
            short_signal = True
        
        # Exit conditions: return to mean
        exit_long = position == 1 and price_close < ma
        exit_short = position == -1 and price_close > ma
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 12h volatility breakout strategy using daily ATR-based bands.
# Enters long when price breaks above daily 20-day mean + 1.5*ATR with volume confirmation (>1.3x avg volume) and price above 50-period EMA.
# Enters short when price breaks below daily 20-day mean - 1.5*ATR with volume confirmation and price below 50-period EMA.
# Uses volatility breakouts to capture momentum moves in both bull and bear markets.
# Volume confirmation ensures institutional participation, EMA filter avoids counter-trend whipsaws.
# Exits when price returns to daily mean, targeting 50-150 trades over 4 years.
# Works in both bull and bear markets by trading breakouts in either direction with trend filter.
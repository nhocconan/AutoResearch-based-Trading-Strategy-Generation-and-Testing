#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for trend and volatility filters
    df_1d = get_htf_data(prices, '1d')
    
    # Daily ATR for volatility filtering
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 12h timeframe indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h ATR for volatility filter
    tr_12h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_12h[0] = high[0] - low[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # 12h EMA20 for trend confirmation
    ema20_12h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(atr_14_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(atr_12h[i]) or 
            np.isnan(ema20_12h[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr_daily = atr_14_aligned[i]
        ema50_daily = ema50_1d_aligned[i]
        atr_12h_val = atr_12h[i]
        ema20_12h_val = ema20_12h[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        price = close[i]
        
        # Volatility filter: daily ATR > 60% of 20-period average (avoid low volatility chop)
        atr_ma_20 = pd.Series(atr_14_aligned).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = atr_daily > 0.6 * atr_ma_20
        
        # Trend filter: price above/below daily EMA50
        uptrend = price > ema50_daily
        downtrend = price < ema50_daily
        
        # Entry conditions with volume confirmation
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long: price breaks above 12h EMA20 + daily uptrend + volatility + volume spike
            if price > ema20_12h_val and uptrend and vol_filter and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h EMA20 + daily downtrend + volatility + volume spike
            elif price < ema20_12h_val and downtrend and vol_filter and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through 12h EMA20 or volatility drops
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on breakdown below EMA20 or volatility collapse
                if price < ema20_12h_val or not vol_filter:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on breakout above EMA20 or volatility collapse
                if price > ema20_12h_val or not vol_filter:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_VolTrend_EMA20_DailyEMA50_ATRFilter"
timeframe = "12h"
leverage = 1.0
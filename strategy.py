#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h ATR for volatility
    tr_12h = np.maximum(high_12h - low_12h,
                       np.maximum(np.abs(high_12h - np.roll(close_12h, 1)),
                                  np.abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # === Daily data (HTF) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 200-day EMA for long-term trend
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === Weekly data (HTF) for regime filter ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 50-week ATR for volatility regime
    tr_1w = np.maximum(high_1w - low_1w,
                       np.maximum(np.abs(high_1w - np.roll(close_1w, 1)),
                                  np.abs(low_1w - np.roll(close_1w, 1))))
    tr_1w[0] = high_1w[0] - low_1w[0]
    atr_1w = pd.Series(tr_1w).rolling(window=50, min_periods=50).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # === 12h Bollinger Bands for mean reversion signals ===
    sma_20_12h = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    std_20_12h = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20_12h + (2.0 * std_20_12h)
    lower_band = sma_20_12h - (2.0 * std_20_12h)
    upper_band_aligned = align_htf_to_ltf(prices, df_12h, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_12h, lower_band)
    
    # === 12h RSI for momentum confirmation ===
    delta = pd.Series(close_12h).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi_values)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_12h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_12h[i]
        atr_12h_val = atr_12h_aligned[i]
        ema_200_val = ema_200_1d_aligned[i]
        atr_1w_val = atr_1w_aligned[i]
        upper_band_val = upper_band_aligned[i]
        lower_band_val = lower_band_aligned[i]
        rsi_val = rsi_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below lower Bollinger Band or RSI < 30
            if price < lower_band_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above upper Bollinger Band or RSI > 70
            if price > upper_band_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Determine market regime based on weekly ATR
            # High volatility regime: use mean reversion
            # Low volatility regime: use trend following
            if atr_1w_val > np.nanmedian(atr_1w_aligned[max(0, i-50):i+1]):
                # High volatility regime: mean reversion at Bollinger Bands
                # LONG: Price touches lower Bollinger Band in uptrend (price > 200-day EMA)
                if (price <= lower_band_val and 
                    price > ema_200_val and 
                    rsi_val < 40):  # Oversold but not extreme
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price touches upper Bollinger Band in downtrend (price < 200-day EMA)
                elif (price >= upper_band_val and 
                      price < ema_200_val and 
                      rsi_val > 60):  # Overbought but not extreme
                    signals[i] = -0.25
                    position = -1
                    continue
            else:
                # Low volatility regime: trend following
                # LONG: Price above 200-day EMA with bullish momentum
                if (price > ema_200_val and 
                    rsi_val > 50 and 
                    rsi_val < 70):  # Not overbought
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price below 200-day EMA with bearish momentum
                elif (price < ema_200_val and 
                      rsi_val < 50 and 
                      rsi_val > 30):  # Not oversold
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_BollingerRSI_TrendMeanReversion"
timeframe = "12h"
leverage = 1.0
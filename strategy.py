#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Calculate daily ATR(14) for volatility regime filter
    daily_tr1 = daily_high - daily_low
    daily_tr2 = np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    daily_tr3 = np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    daily_tr = np.maximum(daily_tr1, np.maximum(daily_tr2, daily_tr3))
    daily_atr_14 = pd.Series(daily_tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Align HTF daily ATR to 6h timeframe
    daily_atr_14_6h = align_htf_to_ltf(prices, df_1d, daily_atr_14)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(daily_atr_14_6h[i]) or np.isnan(atr_14[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: Use daily ATR to distinguish trending vs ranging markets
        # High ATR (> 75th percentile) = trending (favor breakouts)
        # Low ATR (< 25th percentile) = ranging (favor mean reversion)
        # Calculate percentiles using lookback window
        lookback = min(i, 100)  # Use up to 100 periods for percentile
        if lookback >= 20:
            hist_atr = daily_atr_14_6h[max(0, i-lookback):i+1]
            if len(hist_atr) >= 20:
                pct_25 = np.percentile(hist_atr, 25)
                pct_75 = np.percentile(hist_atr, 75)
                current_atr = daily_atr_14_6h[i]
                
                # In trending regime (high volatility): look for breakouts
                if current_atr > pct_75:
                    # 6h breakout conditions with volume confirmation
                    # Calculate 6h Donchian channels (20-period)
                    lookback_6h = min(i, 20)
                    if lookback_6h >= 20:
                        donch_high = np.max(high[i-19:i+1])
                        donch_low = np.min(low[i-19:i+1])
                        
                        # Long: break above Donchian high with volume
                        if (close[i] > donch_high and 
                            volume_ratio[i] > 1.5):
                            signals[i] = 0.25
                        
                        # Short: break below Donchian low with volume
                        elif (close[i] < donch_low and 
                              volume_ratio[i] > 1.5):
                            signals[i] = -0.25
                
                # In ranging regime (low volatility): mean reversion at extremes
                elif current_atr < pct_25:
                    # Calculate 6h RSI(14) for mean reversion signals
                    if i >= 14:
                        # RSI calculation
                        delta = np.diff(close[max(0, i-14):i+1])
                        gain = np.where(delta > 0, delta, 0)
                        loss = np.where(delta < 0, -delta, 0)
                        avg_gain = np.mean(gain) if len(gain) > 0 else 0
                        avg_loss = np.mean(loss) if len(loss) > 0 else 0
                        if avg_loss != 0:
                            rs = avg_gain / avg_loss
                            rsi = 100 - (100 / (1 + rs))
                        else:
                            rsi = 100 if avg_gain > 0 else 50
                        
                        # Long when oversold, short when overbought
                        if rsi < 30 and volume_ratio[i] > 1.2:
                            signals[i] = 0.20
                        elif rsi > 70 and volume_ratio[i] > 1.2:
                            signals[i] = -0.20
    
    return signals

name = "6h_ATR_Regime_Donchian_RSI_MeanReversion"
timeframe = "6h"
leverage = 1.0
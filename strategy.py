#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h mean reversion with 4h trend filter and 1d volatility filter
    # In choppy/mean-reverting markets (2025+), price reverts to VWAP after deviations
    # 4h trend filter ensures we only trade counter-trend in strong trends
    # 1d volatility filter avoids low-volatility chop where mean reversion fails
    # Works in bull/bear: mean reversion occurs in all regimes when volatility is sufficient
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    typical_price = (high + low + close) / 3
    
    # Calculate VWAP (20-period)
    pv = typical_price * volume
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # Calculate standard deviation of price from VWAP (20-period)
    price_dev = typical_price - vwap
    price_dev_sq = price_dev ** 2
    # Manual rolling variance to avoid DataFrame
    var = np.full_like(price_dev_sq, np.nan)
    for i in range(20, len(price_dev_sq)):
        var[i] = np.nanmean(price_dev_sq[i-20:i])
    std_dev = np.sqrt(var)
    
    # Z-score: how many standard deviations price is from VWAP
    zscore = np.divide(price_dev, std_dev, out=np.full_like(price_dev, np.nan), where=std_dev!=0)
    
    # Load 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Load 1d data for volatility filter (ATR ratio)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = high_1d[0] - close_1d[0]
    tr3[0] = low_1d[0] - close_1d[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR (14-period)
    atr = np.full_like(tr, np.nan)
    for i in range(14, len(tr)):
        atr[i] = np.nanmean(tr[i-14:i])
    
    # ATR ratio: current ATR vs 50-period average (volatility filter)
    atr_ma50 = np.full_like(atr, np.nan)
    for i in range(50, len(atr)):
        atr_ma50[i] = np.nanmean(atr[i-50:i])
    atr_ratio = np.divide(atr, atr_ma50, out=np.full_like(atr, np.nan), where=atr_ma50!=0)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(zscore[i]) or 
            np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(atr_ratio[i]) or
            hours[i] < 8 or hours[i] > 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price significantly below VWAP (zscore < -2) in uptrend (price > EMA50_4h)
            # Only trade mean reversion when volatility is sufficient (atr_ratio > 0.8)
            if zscore[i] < -2.0 and close[i] > ema50_4h_aligned[i] and atr_ratio[i] > 0.8:
                signals[i] = 0.20
                position = 1
            # Short: Price significantly above VWAP (zscore > 2) in downtrend (price < EMA50_4h)
            elif zscore[i] > 2.0 and close[i] < ema50_4h_aligned[i] and atr_ratio[i] > 0.8:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: Return to VWAP (zscore crosses zero) or opposite extreme
            if position == 1:
                if zscore[i] > 0:  # Price crossed back above VWAP
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if zscore[i] < 0:  # Price crossed back below VWAP
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_VWAP_MeanReversion_4hEMA50_Trend_1dATRRatio_Filter_v1"
timeframe = "1h"
leverage = 1.0
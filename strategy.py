#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h timeframe with 4h ADX trend filter and 1d volatility filter
    # Uses 4h ADX > 25 to identify trending markets, 1d ATR-based volatility filter
    # Enter long/short on 1h EMA(9)/EMA(21) crossovers during strong trends
    # Works in bull/bear: captures trend momentum while avoiding choppy markets
    
    # Load 4h data for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX(14) on 4h
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            elif minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
                
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        for i in range(period, len(high)):
            if atr[i] != 0:
                plus_di[i] = 100 * (np.mean(plus_dm[i-period+1:i+1]) / atr[i])
                minus_di[i] = 100 * (np.mean(minus_dm[i-period+1:i+1]) / atr[i])
        
        dx = np.zeros_like(high)
        for i in range(period, len(high)):
            if (plus_di[i] + minus_di[i]) != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Load 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on 1d
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Price and EMA data on 1h
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA(9) and EMA(21) for crossover signals
    ema_9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Session filter: 8-20 UTC (already datetime64)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after EMA warmup
        # Skip if not in session or data not ready
        if not in_session[i] or np.isnan(adx_4h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(ema_9[i]) or np.isnan(ema_21[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 4h ADX > 25 indicates strong trend
        # Volatility filter: 1d ATR > 20-period average indicates sufficient volatility
        atr_ma20 = np.zeros_like(atr_1d_aligned)
        if i >= 20:
            atr_ma20[i] = np.mean(atr_1d_aligned[i-20:i])
        vol_filter = atr_1d_aligned[i] > atr_ma20[i] if i >= 20 else False
        
        if position == 0:
            # Long: EMA(9) crosses above EMA(21) in strong trend with sufficient volatility
            if ema_9[i] > ema_21[i] and ema_9[i-1] <= ema_21[i-1] and adx_4h_aligned[i] > 25 and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: EMA(9) crosses below EMA(21) in strong trend with sufficient volatility
            elif ema_9[i] < ema_21[i] and ema_9[i-1] >= ema_21[i-1] and adx_4h_aligned[i] > 25 and vol_filter:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: EMA(9) crosses back in opposite direction
            if position == 1:
                if ema_9[i] < ema_21[i] and ema_9[i-1] >= ema_21[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if ema_9[i] > ema_21[i] and ema_9[i-1] <= ema_21[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_ADX25_VolFilter_EMA9_21_Crossover_v1"
timeframe = "1h"
leverage = 1.0
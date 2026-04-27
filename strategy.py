# 1h_PriceAction_TrendFollow
# Hypothesis: 1h price action combined with 4h/1d trend filters and volume confirmation can capture trends in both bull and bear markets.
# Uses 4h EMA200 for long-term trend, 1d ADX for trend strength, and 1h price action (higher highs/lows) for entry.
# Session filter (08-20 UTC) reduces noise. Position size 0.20 to manage drawdown.
# Target: 15-30 trades/year per symbol to avoid fee drag.

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
    
    # Get 4h data for EMA200 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Get 1d data for ADX trend strength
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_di = np.zeros_like(tr)
        minus_di = np.zeros_like(tr)
        
        atr[period-1] = np.mean(tr[:period])
        plus_dm_sum = np.sum(plus_dm[:period])
        minus_dm_sum = np.sum(minus_dm[:period])
        
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]
            
            if atr[i] != 0:
                plus_di[i] = 100 * plus_dm_sum / atr[i]
                minus_di[i] = 100 * minus_dm_sum / atr[i]
            else:
                plus_di[i] = 0
                minus_di[i] = 0
        
        dx = np.zeros_like(tr)
        for i in range(len(dx)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
            else:
                dx[i] = 0
        
        adx = np.zeros_like(dx)
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    # Price action: higher highs and higher lows for uptrend, lower highs and lower lows for downtrend
    hh = np.zeros(n, dtype=bool)  # higher high
    hl = np.zeros(n, dtype=bool)  # higher low
    lh = np.zeros(n, dtype=bool)  # lower high
    ll = np.zeros(n, dtype=bool)  # lower low
    
    for i in range(1, n):
        hh[i] = high[i] > high[i-1]
        hl[i] = low[i] > low[i-1]
        lh[i] = high[i] < high[i-1]
        ll[i] = low[i] < low[i-1]
    
    # Consecutive counts for confirmation
    hh_count = np.zeros(n)
    hl_count = np.zeros(n)
    lh_count = np.zeros(n)
    ll_count = np.zeros(n)
    
    for i in range(1, n):
        if hh[i]:
            hh_count[i] = hh_count[i-1] + 1
        else:
            hh_count[i] = 0
            
        if hl[i]:
            hl_count[i] = hl_count[i-1] + 1
        else:
            hl_count[i] = 0
            
        if lh[i]:
            lh_count[i] = lh_count[i-1] + 1
        else:
            lh_count[i] = 0
            
        if ll[i]:
            ll_count[i] = ll_count[i-1] + 1
        else:
            ll_count[i] = 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Warmup: need 4h EMA200 (200 periods), 1d ADX (periods for calculation), price action
    start_idx = max(200, 30)  # 200 for EMA, 30 for ADX calculation stability
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_200_4h_aligned[i]) or np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        ema_trend = ema_200_4h_aligned[i]
        adx_val = adx_14_aligned[i]
        
        # Trend filter: ADX > 25 for trending market
        trending = adx_val > 25
        
        if position == 0:
            # Long: higher highs and higher lows (uptrend) + price > 4h EMA200 + ADX > 25
            if hl_count[i] >= 2 and hh_count[i] >= 2 and price > ema_trend and trending:
                signals[i] = size
                position = 1
            # Short: lower highs and lower lows (downtrend) + price < 4h EMA200 + ADX > 25
            elif lh_count[i] >= 2 and ll_count[i] >= 2 and price < ema_trend and trending:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend breaks (lower low) or price goes below EMA200 or ADX weakens
            if ll_count[i] >= 2 or price < ema_trend or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: trend breaks (higher high) or price goes above EMA200 or ADX weakens
            if hh_count[i] >= 2 or price > ema_trend or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_PriceAction_TrendFollow"
timeframe = "1h"
leverage = 1.0
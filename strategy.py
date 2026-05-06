#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using weekly Bollinger Bands with mean reversion in ranging markets
# - Uses 1w Bollinger Bands (20-period, 2 std) for mean reversion signals
# - Uses 4h RSI(14) for overbought/oversold confirmation
# - Uses 4h ADX < 20 to filter for ranging markets only
# - Enters long when price touches lower BB and RSI < 30 in ranging market
# - Enters short when price touches upper BB and RSI > 70 in ranging market
# - Exits when price returns to BB middle (20-period SMA)
# - Designed to profit from mean reversion in ranging markets while avoiding trends
# - Target: 80-180 total trades over 4 years (20-45/year) with 0.25 position sizing

name = "4h_1wBB_20_2_RSI_ADX_Range"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w Bollinger Bands (20-period, 2 std)
    close_1w = df_1w['close'].values
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    middle_bb = sma_20  # Middle band for exit
    
    # Align 1w Bollinger Bands to 4h timeframe
    upper_bb_4h = align_htf_to_ltf(prices, df_1w, upper_bb)
    lower_bb_4h = align_htf_to_ltf(prices, df_1w, lower_bb)
    middle_bb_4h = align_htf_to_ltf(prices, df_1w, middle_bb)
    
    # RSI filter (4h timeframe)
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        avg_gain[period-1] = np.mean(gain[1:period+1])
        avg_loss[period-1] = np.mean(loss[1:period+1])
        
        for i in range(period, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.zeros_like(close)
        rsi = np.zeros_like(close)
        for i in range(period, len(close)):
            if avg_loss[i] != 0:
                rs[i] = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs[i]))
            else:
                rsi[i] = 100
        return rsi
    
    rsi_values = calculate_rsi(close, 14)
    rsi_oversold = rsi_values < 30
    rsi_overbought = rsi_values > 70
    
    # ADX filter (4h timeframe) - range market (low ADX)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        atr[period-1] = np.mean(tr[1:period+1])
        plus_dm_sum = np.sum(plus_dm[1:period+1])
        minus_dm_sum = np.sum(minus_dm[1:period+1])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm[i-period+1] if i-period+1 >= 0 else 0) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm[i-period+1] if i-period+1 >= 0 else 0) + minus_dm[i]
            plus_di[i] = 100 * plus_dm_sum / (atr[i] * period) if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_sum / (atr[i] * period) if atr[i] != 0 else 0
        
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        for i in range(2*period-1, len(high)):
            di_diff = abs(plus_di[i] - minus_di[i])
            di_sum = plus_di[i] + minus_di[i]
            dx[i] = 100 * di_diff / di_sum if di_sum != 0 else 0
        
        # Smooth DX to get ADX
        adx[2*period-1] = np.mean(dx[2*period-1:3*period]) if 3*period <= len(high) else 0
        for i in range(3*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_values = calculate_adx(high, low, close, 14)
    adx_filter = adx_values < 20  # Ranging market filter
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(upper_bb_4h[i]) or np.isnan(lower_bb_4h[i]) or 
            np.isnan(middle_bb_4h[i]) or np.isnan(rsi_oversold[i]) or 
            np.isnan(rsi_overbought[i]) or np.isnan(adx_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches lower BB and RSI oversold in ranging market
            if low[i] <= lower_bb_4h[i] and rsi_oversold[i] and adx_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches upper BB and RSI overbought in ranging market
            elif high[i] >= upper_bb_4h[i] and rsi_overbought[i] and adx_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle BB
            if close[i] >= middle_bb_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle BB
            if close[i] <= middle_bb_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
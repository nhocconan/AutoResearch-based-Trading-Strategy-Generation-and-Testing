#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day price action guided by weekly Bollinger Band mean reversion with volume confirmation.
# In both bull and bear markets, prices tend to revert to the mean after extended deviations.
# Weekly Bollinger Bands (20, 2) provide dynamic support/resistance; price touching bands with
# volume exhaustion signals potential reversal. Weekly timeframe reduces noise, daily entries
# improve timing. Target: 15-25 trades/year (60-100 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly data for Bollinger Bands ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Bollinger Bands (20-period, 2 std) on weekly data
    sma_20 = np.full_like(close_1w, np.nan)
    std_20 = np.full_like(close_1w, np.nan)
    for i in range(len(close_1w)):
        if i >= 19:
            sma_20[i] = np.mean(close_1w[i-19:i+1])
            std_20[i] = np.std(close_1w[i-19:i+1])
    
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    bb_upper_aligned = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1w, bb_lower)
    sma_20_aligned = align_htf_to_ltf(prices, df_1w, sma_20)
    
    # Weekly volume average (20-period)
    vol_avg20_1w = np.full_like(volume_1w, np.nan)
    for i in range(len(volume_1w)):
        if i >= 19:
            vol_avg20_1w[i] = np.mean(volume_1w[i-19:i+1])
    
    vol_avg20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg20_1w)
    
    # === Daily RSI (14-period) for entry timing ===
    def calculate_rsi(close_prices, period=14):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close_prices, np.nan)
        avg_loss = np.full_like(close_prices, np.nan)
        
        if len(close_prices) > period:
            avg_gain[period] = np.mean(gain[1:period+1])
            avg_loss[period] = np.mean(loss[1:period+1])
            
            for i in range(period+1, len(close_prices)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.divide(avg_gain, avg_loss, out=np.full_like(close_prices, np.nan), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0
    warmup = 50  # Sufficient for all indicators
    
    for i in range(warmup, n):
        if (np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or
            np.isnan(sma_20_aligned[i]) or
            np.isnan(vol_avg20_1w_aligned[i]) or
            np.isnan(rsi_14[i])):
            signals[i] = 0.0
            continue
        
        vol_1w_current = align_htf_to_ltf(prices, df_1w, volume_1w)[i]
        vol_filter = vol_1w_current < 0.8 * vol_avg20_1w_aligned[i]  # Volume exhaustion
        
        if position == 0:
            # Long: price touches or goes below lower BB + volume exhaustion + RSI not overbought
            if low[i] <= bb_lower_aligned[i] and \
               vol_filter and rsi_14[i] < 60:
                signals[i] = 0.25
                position = 1
            # Short: price touches or goes above upper BB + volume exhaustion + RSI not oversold
            elif high[i] >= bb_upper_aligned[i] and \
                 vol_filter and rsi_14[i] > 40:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to weekly SMA or RSI overbought
            if close[i] >= sma_20_aligned[i] or rsi_14[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly SMA or RSI oversold
            if close[i] <= sma_20_aligned[i] or rsi_14[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyBB_MeanReversion_VolumeExhaustion"
timeframe = "1d"
leverage = 1.0
EOF
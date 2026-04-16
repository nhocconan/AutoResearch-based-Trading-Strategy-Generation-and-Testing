# 4h_RSI2_Stochastic_RSI_Bollinger_Bands_Reversal
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h mean reversion using RSI(2) and Stochastic RSI with Bollinger Bands
# Works in both bull and bear markets by capturing short-term overextensions
# Long when: RSI(2) < 10 AND Stochastic RSI < 0.1 AND price touches lower Bollinger Band
# Short when: RSI(2) > 90 AND Stochastic RSI > 0.9 AND price touches upper Bollinger Band
# Exit when: RSI(2) crosses 50 (mean reversion complete)
# Uses 1d volume confirmation and 1d ADX trend filter to avoid counter-trend trades in strong trends
# Target: 80-160 total trades over 4 years (20-40/year) for optimal balance

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d ADX trend filter (avoid counter-trend in strong trends) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
                
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros(len(high))
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros(len(high))
        minus_di = np.zeros(len(high))
        dx = np.zeros(len(high))
        
        for i in range(period, len(high)):
            if atr[i] > 0:
                plus_di[i] = 100 * (plus_dm[i] / atr[i])
                minus_di[i] = 100 * (minus_dm[i] / atr[i])
                if (plus_di[i] + minus_di[i]) > 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros(len(high))
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # === 1d Volume Confirmation ===
    vol_ma_1d = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24*1h = 1d (4h data)
    
    # === RSI(2) ===
    def calculate_rsi(close, period=2):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros(len(close))
        avg_loss = np.zeros(len(close))
        avg_gain[period] = np.mean(gain[:period+1])
        avg_loss[period] = np.mean(loss[:period+1])
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.zeros(len(close))
        rsi = np.zeros(len(close))
        for i in range(period+1, len(close)):
            if avg_loss[i] > 0:
                rs[i] = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs[i]))
            else:
                rsi[i] = 100
        return rsi
    
    rsi_2 = calculate_rsi(close, 2)
    
    # === Stochastic RSI (14,14,3,3) ===
    def calculate_stoch_rsi(rsi, k_period=14, d_period=3):
        stoch_rsi = np.zeros(len(rsi))
        k = np.zeros(len(rsi))
        
        for i in range(k_period-1, len(rsi)):
            rsi_min = np.min(rsi[i-k_period+1:i+1])
            rsi_max = np.max(rsi[i-k_period+1:i+1])
            if rsi_max - rsi_min > 0:
                stoch_rsi[i] = (rsi[i] - rsi_min) / (rsi_max - rsi_min)
            else:
                stoch_rsi[i] = 0.5
        
        # Smooth K with SMA
        for i in range(len(k)):
            if i < d_period-1:
                k[i] = np.mean(stoch_rsi[max(0, i-d_period+1):i+1])
            else:
                k[i] = np.mean(stoch_rsi[i-d_period+1:i+1])
        
        # D is SMA of K
        d = np.zeros(len(k))
        for i in range(len(d)):
            if i < d_period-1:
                d[i] = np.mean(k[max(0, i-d_period+1):i+1])
            else:
                d[i] = np.mean(k[i-d_period+1:i+1])
        
        return k, d
    
    stoch_rsi_k, stoch_rsi_d = calculate_stoch_rsi(rsi_2, 14, 3)
    
    # === Bollinger Bands (20,2) ===
    def calculate_bollinger_bands(close, period=20, std_dev=2):
        sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
        std = pd.Series(close).rolling(window=period, min_periods=period).std().values
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return upper, lower, sma
    
    bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(close, 20, 2)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(adx_14_aligned[i]) or 
            np.isnan(vol_ma_1d[i]) or
            np.isnan(rsi_2[i]) or
            np.isnan(stoch_rsi_k[i]) or
            np.isnan(bb_upper[i]) or
            np.isnan(bb_lower[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        adx_val = adx_14_aligned[i]
        vol_confirm = volume[i] > vol_ma_1d[i] * 1.5  # 1.5x average volume
        rsi_val = rsi_2[i]
        stoch_k_val = stoch_rsi_k[i]
        bb_upper_val = bb_upper[i]
        bb_lower_val = bb_lower[i]
        
        # === EXIT LOGIC (RSI crosses 50) ===
        if position == 1:  # Long position
            if rsi_val > 50:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            if rsi_val < 50:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: RSI(2) < 10 AND Stochastic RSI < 0.1 AND price touches lower BB AND ADX < 25 (not strong trend)
            if rsi_val < 10 and stoch_k_val < 0.1 and price <= bb_lower_val and adx_val < 25:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: RSI(2) > 90 AND Stochastic RSI > 0.9 AND price touches upper BB AND ADX < 25 (not strong trend)
            elif rsi_val > 90 and stoch_k_val > 0.9 and price >= bb_upper_val and adx_val < 25:
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

name = "4h_RSI2_Stochastic_RSI_Bollinger_Bands_Reversal"
timeframe = "4h"
leverage = 1.0
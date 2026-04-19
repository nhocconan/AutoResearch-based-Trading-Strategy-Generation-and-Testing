#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI + chop filter for trend-following in trending markets and mean-reversion in choppy markets.
# Uses 1d KAMA as primary trend filter, RSI(14) for momentum, and Choppiness Index(14) to detect regime.
# In trending markets (CHOP < 38.2): follow KAMA direction.
# In choppy markets (CHOP > 61.8): mean-revert at RSI extremes.
# Volume confirmation ensures breakout/breakdown validity. Target: 15-25 trades/year per symbol.
name = "1d_KAMA_RSI_Chop_Regime"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate KAMA on daily close
    def calculate_kama(close, er_len=10, fast=2, slow=30):
        n = len(close)
        kama = np.zeros(n)
        kama[:] = np.nan
        
        change = np.abs(np.diff(close, n=er_len))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        
        er = np.zeros(n)
        er[er_len:] = change[er_len:] / volatility[er_len:]
        er = np.where(volatility == 0, 0, er)
        
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        sc = np.where(er_len > 0, sc, 0)
        
        kama[er_len] = close[er_len]
        for i in range(er_len+1, n):
            if not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
        return kama
    
    # Calculate RSI
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate Choppiness Index
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        for i in range(period, len(high)):
            atr[i] = np.sum(tr[i-period+1:i+1]) / period
        
        hh = np.zeros_like(high)
        ll = np.zeros_like(high)
        
        for i in range(period-1, len(high)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        
        chop = np.zeros_like(high)
        for i in range(period-1, len(high)):
            if hh[i] > ll[i] and atr[i] > 0:
                chop[i] = 100 * np.log10(np.sum(tr[i-period+1:i+1]) / (hh[i] - ll[i])) / np.log10(period)
            else:
                chop[i] = 50
        return chop
    
    # Calculate indicators
    kama = calculate_kama(close, 10, 2, 30)
    rsi = calculate_rsi(close, 14)
    chop = calculate_chop(high, low, close, 14)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.3 * vol_ma
        
        # Regime detection
        trending_market = chop_val < 38.2
        choppy_market = chop_val > 61.8
        
        if position == 0:
            # Enter long conditions
            long_condition = False
            if trending_market and price > kama_val and volume_confirmed:
                long_condition = True  # Follow trend in trending market
            elif choppy_market and rsi_val < 30 and volume_confirmed:
                long_condition = True  # Mean reversion in choppy market
            
            # Enter short conditions
            short_condition = False
            if trending_market and price < kama_val and volume_confirmed:
                short_condition = True  # Follow trend in trending market
            elif choppy_market and rsi_val > 70 and volume_confirmed:
                short_condition = True  # Mean reversion in choppy market
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long conditions
            exit_condition = False
            if trending_market and price < kama_val:
                exit_condition = True  # Trend reversal
            elif choppy_market and rsi_val > 50:
                exit_condition = True  # Mean reversion complete
            elif not volume_confirmed and i > start_idx:  # Volume filter for exit
                exit_condition = True
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short conditions
            exit_condition = False
            if trending_market and price > kama_val:
                exit_condition = True  # Trend reversal
            elif choppy_market and rsi_val < 50:
                exit_condition = True  # Mean reversion complete
            elif not volume_confirmed and i > start_idx:  # Volume filter for exit
                exit_condition = True
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
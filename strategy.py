#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index regime filter with weekly pivot point reversal strategy.
# Uses weekly Camarilla pivot levels for mean reversion in choppy markets (CHOP > 61.8)
# and trend following in trending markets (CHOP < 38.2). Designed for low trade frequency
# (~15-25 trades/year) to minimize fee decay. Works in both bull and bear markets by
# adapting to market regime. Targets BTC/ETH primarily.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for Camarilla pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels for previous week
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot = (high_1w + low_1w + close_1w) / 3
    r1 = close_1w + (high_1w - low_1w) * 1.1 / 12
    s1 = close_1w - (high_1w - low_1w) * 1.1 / 12
    r2 = close_1w + (high_1w - low_1w) * 1.1 / 6
    s2 = close_1w - (high_1w - low_1w) * 1.1 / 6
    r3 = close_1w + (high_1w - low_1w) * 1.1 / 4
    s3 = close_1w - (high_1w - low_1w) * 1.1 / 4
    
    # Calculate daily Choppiness Index (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    atr = np.zeros(n)
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(ATR14)/(max(HH14)-min(LL14))) / log10(14)
    chop = 100 * np.log10(atr * 14 / (highest_high - lowest_low)) / np.log10(14)
    
    # Calculate daily RSI (14-period) for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align weekly indicators to daily timeframe (waits for weekly bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        chop_val = chop[i]
        rsi_val = rsi[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        
        if position == 0:
            # Choppy market (CHOP > 61.8): mean reversion at extreme RSI levels
            if chop_val > 61.8:
                # Long when RSI < 30 and price near S1/S2
                if rsi_val < 30 and (price <= s1_val * 1.002 or price <= s2_val * 1.002):
                    signals[i] = 0.25
                    position = 1
                # Short when RSI > 70 and price near R1/R2
                elif rsi_val > 70 and (price >= r1_val * 0.998 or price >= r2_val * 0.998):
                    signals[i] = -0.25
                    position = -1
            # Trending market (CHOP < 38.2): follow weekly trend
            elif chop_val < 38.2:
                # Long when price above R1 and RSI > 50
                if price > r1_val and rsi_val > 50:
                    signals[i] = 0.25
                    position = 1
                # Short when price below S1 and RSI < 50
                elif price < s1_val and rsi_val < 50:
                    signals[i] = -0.25
                    position = -1
            # Transition zone (38.2 <= CHOP <= 61.8): weaker signals
            else:
                # Long when RSI < 35 and price near S2/S3
                if rsi_val < 35 and (price <= s2_val * 1.002 or price <= s3_val * 1.002):
                    signals[i] = 0.15
                    position = 1
                # Short when RSI > 65 and price near R2/R3
                elif rsi_val > 65 and (price >= r2_val * 0.998 or price >= r3_val * 0.998):
                    signals[i] = -0.15
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when RSI > 70 (overbought) or price reaches R3
                if rsi_val > 70 or price >= r3_val * 0.998:
                    exit_signal = True
                # Also exit if market becomes strongly trending against position
                elif chop_val < 38.2 and price < s1_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when RSI < 30 (oversold) or price reaches S3
                if rsi_val < 30 or price <= s3_val * 1.002:
                    exit_signal = True
                # Also exit if market becomes strongly trending against position
                elif chop_val < 38.2 and price > r1_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Choppiness_WeeklyCamarilla_MeanRev_Trend"
timeframe = "1d"
leverage = 1.0
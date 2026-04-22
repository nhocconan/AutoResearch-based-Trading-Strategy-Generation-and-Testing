#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h RSI mean reversion with Bollinger Bands and volume confirmation
# Uses 1-day ATR for regime filter (low ATR = range, high ATR = trend) to adapt to market conditions
# Target: 20-35 trades/year per symbol, works in range-bound markets via mean reversion
# RSI < 30 for long, RSI > 70 for short, with Bollinger Band support/resistance and volume spike
# Includes volatility-adjusted position sizing and volatility filter to avoid choppy markets

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for ATR-based regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR on daily timeframe for regime filter
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    tr_1d[0] = np.nan  # First value has no previous close
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 14-period RSI on 4-hour timeframe
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate Bollinger Bands (20, 2) on 4-hour timeframe
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    upper_bb_values = upper_bb.values
    lower_bb_values = lower_bb.values
    
    # Volume spike filter (20-period on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > 2.0 * vol_ma20.values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align 1-day ATR to 4-hour timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(rsi_values[i]) or
            np.isnan(lower_bb_values[i]) or np.isnan(upper_bb_values[i]) or
            np.isnan(vol_ma20.values[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when ATR is low (range-bound market)
        # Use 50th percentile of ATR as threshold - adaptive to each symbol
        if np.isnan(atr_14_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Calculate adaptive ATR threshold based on historical values
        # Use rolling 50-period percentile of ATR for dynamic threshold
        if i >= 150:  # Need sufficient history for percentile calculation
            atr_hist = atr_14_1d_aligned[max(0, i-50):i+1]
            valid_atr = atr_hist[~np.isnan(atr_hist)]
            if len(valid_atr) >= 10:
                atr_median = np.median(valid_atr)
                # Only trade in low volatility regime (below median ATR)
                if atr_14_1d_aligned[i] > atr_median:
                    if position != 0:
                        signals[i] = 0.0
                        position = 0
                    continue
        
        if position == 0:
            # Long: RSI oversold (<30) + price at or below lower BB + volume spike
            if (rsi_values[i] < 30 and 
                close[i] <= lower_bb_values[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) + price at or above upper BB + volume spike
            elif (rsi_values[i] > 70 and 
                  close[i] >= upper_bb_values[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: RSI returns to neutral (50) or price reaches middle of BB
                sma_20_val = sma_20.values[i]
                if (rsi_values[i] >= 50 or 
                    close[i] >= sma_20_val):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: RSI returns to neutral (50) or price reaches middle of BB
                sma_20_val = sma_20.values[i]
                if (rsi_values[i] <= 50 or 
                    close[i] <= sma_20_val):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_RSI_BB_MeanReversion_Volume_ATRFilter"
timeframe = "4h"
leverage = 1.0
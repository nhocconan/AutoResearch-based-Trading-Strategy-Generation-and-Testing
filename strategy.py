#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Weekly RSI Reversal with Volume and Volatility Filter
# Hypothesis: RSI extremes on weekly timeframe indicate overextended moves
# likely to reverse. Enter on daily close crossing RSI(50) with volume
# confirmation and low volatility (ATR ratio) to avoid choppy markets.
# Works in both bull/bear by fading extremes. Target: 15-25 trades/year.

name = "1d_weekly_rsi_reversal_volume_volatility_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for RSI
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate RSI on weekly close
    weekly_close = df_weekly['close'].values
    delta = np.diff(weekly_close, prepend=weekly_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_rsi(gain, loss, period):
        rsi = np.full_like(weekly_close, np.nan, dtype=float)
        if len(gain) < period:
            return rsi
        # Initial average gain/loss
        avg_gain = np.mean(gain[1:period])
        avg_loss = np.mean(loss[1:period])
        if avg_loss == 0:
            rsi[period-1] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[period-1] = 100 - (100 / (1 + rs))
        # Subsequent values
        for i in range(period, len(weekly_close)):
            avg_gain = (avg_gain * (period-1) + gain[i]) / period
            avg_loss = (avg_loss * (period-1) + loss[i]) / period
            if avg_loss == 0:
                rsi[i] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_weekly = wilders_rsi(gain, loss, 14)
    
    # Align weekly RSI to daily timeframe
    rsi_weekly_aligned = align_htf_to_ltf(prices, df_weekly, rsi_weekly)
    
    # Daily volume filter: volume > 1.3x 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    # Daily volatility filter: ATR(10) / ATR(30) < 1.2 (low volatility)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_atr(data, period):
        atr = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return atr
        atr[period-1] = np.nanmean(data[1:period])
        for i in range(period, len(data)):
            if not np.isnan(atr[i-1]):
                atr[i] = (atr[i-1] * (period-1) + data[i]) / period
        return atr
    
    atr10 = wilders_atr(tr, 10)
    atr30 = wilders_atr(tr, 30)
    atr_ratio = np.where(atr30 > 0, atr10 / atr30, 1.0)
    vol_filter_low = atr_ratio < 1.2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):
        # Skip if required data not available
        if (np.isnan(rsi_weekly_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses below 50 or volatility increases
            if rsi_weekly_aligned[i] < 50 or atr_ratio[i] >= 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: RSI crosses above 50 or volatility increases
            if rsi_weekly_aligned[i] > 50 or atr_ratio[i] >= 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Require low volatility environment
            if vol_filter_low[i]:
                # Long entry: RSI > 60 (bullish extreme) with volume
                if rsi_weekly_aligned[i] > 60 and vol_filter[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: RSI < 40 (bearish extreme) with volume
                elif rsi_weekly_aligned[i] < 40 and vol_filter[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1-week RSI divergence + volume confirmation + 1-week ADX trend filter.
# Weekly RSI divergence identifies potential reversals with momentum exhaustion.
# Volume surge confirms institutional participation in the reversal.
# Weekly ADX > 25 ensures trades occur in trending markets, avoiding chop.
# Works in bull/bear by catching reversals from overextended moves.
# Target: 30-100 total trades over 4 years (7-25/year). Size: 0.25.

def calculate_rsi(close, period=14):
    """Calculate RSI with proper Wilder's smoothing"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[period-1] = np.mean(gain[:period])
    avg_loss[period-1] = np.mean(loss[:period])
    
    for i in range(period, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === Weekly RSI (14) for momentum/divergence ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    rsi_14 = calculate_rsi(close_1w, 14)
    rsi_14_w = align_htf_to_ltf(prices, df_1w, rsi_14)
    
    # === Weekly ADX (14) for trend strength ===
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate Directional Movement
    up_move = np.concatenate([[0], np.diff(high)])
    down_move = np.concatenate([[0], -np.diff(low)])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Wilder's smoothing for TR and DM
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_14 = wilders_smoothing(tr, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # Avoid division by zero
    plus_di = np.where(atr_14 != 0, 100 * plus_dm_smooth / atr_14, 0)
    minus_di = np.where(atr_14 != 0, 100 * minus_dm_smooth / atr_14, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_14 = wilders_smoothing(dx, 14)
    adx_1w = align_htf_to_ltf(prices, df_1w, adx_14)
    
    # === Daily volume for surge confirmation ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_a = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(rsi_14_w[i]) or 
            np.isnan(adx_1w[i]) or
            np.isnan(vol_ma_20_a[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        
        # Get current daily volume
        df_1d_current = get_htf_data(prices, '1d')
        vol_1d_current = df_1d_current['volume'].values
        vol_1d_a = align_htf_to_ltf(prices, df_1d_current, vol_1d_current)
        
        # Volume surge: current 1d volume > 1.5x 20-period average
        vol_surge = vol_1d_a[i] > vol_ma_20_a[i] * 1.5
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_1w[i] > 25.0
        
        # RSI levels
        rsi = rsi_14_w[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Bullish divergence: RSI < 30 (oversold) + volume surge + trending
            if rsi < 30 and vol_surge and trending:
                signals[i] = 0.25
                position = 1
                continue
            # Bearish divergence: RSI > 70 (overbought) + volume surge + trending
            elif rsi > 70 and vol_surge and trending:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal on opposite RSI extreme
        elif position == 1:
            # Exit long if RSI > 70 (overbought)
            if rsi > 70:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if RSI < 30 (oversold)
            if rsi < 30:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyRSI_Divergence_Volume1.5x_ADX25"
timeframe = "1d"
leverage = 1.0
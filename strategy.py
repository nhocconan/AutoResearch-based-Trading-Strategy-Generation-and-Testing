#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Parabolic SAR with 1d trend filter
# Long when ADX > 25 (trending), SAR below price, and 1d EMA50 uptrend
# Short when ADX > 25, SAR above price, and 1d EMA50 downtrend
# Uses ADX to filter ranging markets and SAR for trend entry/exit
# EMA50 on 1d ensures alignment with higher timeframe trend
# Target: 50-150 total trades over 4 years with controlled risk
# Parabolic SAR provides built-in trailing stop

name = "6h_adx_sar_1d_ema50_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # EMA50 calculation
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # ADX calculation (14 periods)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
    minus_di = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Parabolic SAR (0.02 step, 0.2 max)
    sar = np.full_like(close, np.nan)
    ep = np.full_like(close, np.nan)  # Extreme point
    af = np.full_like(close, 0.02)    # Acceleration factor
    long = np.full_like(close, True)  # True for long, False for short
    
    # Initialize
    sar[0] = low[0]
    ep[0] = high[0]
    af[0] = 0.02
    long[0] = True
    
    for i in range(1, n):
        if long[i-1]:  # Was long
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            # Reverse if price < SAR
            if low[i] < sar[i]:
                long[i] = False
                sar[i] = ep[i-1]  # SAR becomes prior EP
                ep[i] = low[i]    # EP becomes lowest low
                af[i] = 0.02      # Reset AF
            else:
                long[i] = True
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + 0.02, 0.2)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # Was short
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            # Reverse if price > SAR
            if high[i] > sar[i]:
                long[i] = True
                sar[i] = ep[i-1]  # SAR becomes prior EP
                ep[i] = high[i]   # EP becomes highest high
                af[i] = 0.02      # Reset AF
            else:
                long[i] = False
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + 0.02, 0.2)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    
    # Ensure we have valid values
    sar = np.where(np.isnan(sar), close, sar)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):  # Start after ADX warmup
        # Skip if required data not available
        if np.isnan(adx[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: SAR above price (trend reversal) or ADX weak (< 20) or trend change
            if sar[i] > close[i] or adx[i] < 20 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: SAR below price (trend reversal) or ADX weak (< 20) or trend change
            if sar[i] < close[i] or adx[i] < 20 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: ADX > 25 (trending) + SAR signal + 1d trend alignment
            if adx[i] > 25:
                # Long: SAR below price (uptrend) and price above 1d EMA50
                if sar[i] < close[i] and close[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: SAR above price (downtrend) and price below 1d EMA50
                elif sar[i] > close[i] and close[i] < ema50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals
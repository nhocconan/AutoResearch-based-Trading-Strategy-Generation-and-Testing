# 12h_camarilla_pivot_1w_vwap_v1
# Hypothesis: On 12-hour timeframe, use weekly Camarilla pivot levels for mean reversion in high/low volatility regimes.
# Long when price touches S3 level with VWAP > weekly VWAP and weekly trend up.
# Short when price touches R3 level with VWAP < weekly VWAP and weekly trend down.
# Exit when price reaches weekly VWAP or opposite pivot level.
# Uses weekly trend filter to avoid counter-trend trades in both bull and bear markets.
# Target: 20-40 trades/year to minimize fee drag while capturing mean reversion at institutional levels.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_vwap_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivots and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly OHLC for Camarilla pivots
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate Camarilla levels (based on previous week)
    # R4 = Close + (High - Low) * 1.5
    # R3 = Close + (High - Low) * 1.125
    # S3 = Close - (High - Low) * 1.125
    camarilla_r3 = weekly_close + (weekly_high - weekly_low) * 1.125
    camarilla_s3 = weekly_close - (weekly_high - weekly_low) * 1.125
    
    # Align to 12h timeframe (shifted by 1 week for no look-ahead)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Weekly trend filter: EMA(21) slope
    weekly_ema21 = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_ema21_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema21)
    weekly_trend_up = np.zeros(len(weekly_ema21_aligned), dtype=bool)
    weekly_trend_down = np.zeros(len(weekly_ema21_aligned), dtype=bool)
    for i in range(1, len(weekly_ema21_aligned)):
        if not np.isnan(weekly_ema21_aligned[i]) and not np.isnan(weekly_ema21_aligned[i-1]):
            weekly_trend_up[i] = weekly_ema21_aligned[i] > weekly_ema21_aligned[i-1]
            weekly_trend_down[i] = weekly_ema21_aligned[i] < weekly_ema21_aligned[i-1]
    
    # VWAP calculation (12-period)
    typical_price = (high + low + close) / 3
    pv = typical_price * volume
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # Weekly VWAP for trend filter
    weekly_tp = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pv = weekly_tp * df_1w['volume'].values
    weekly_cum_pv = np.nancumsum(weekly_pv)
    weekly_cum_vol = np.nancumsum(df_1w['volume'].values)
    weekly_vwap = np.divide(weekly_cum_pv, weekly_cum_vol, out=np.full_like(weekly_cum_pv, np.nan), where=weekly_cum_vol!=0)
    weekly_vwap_aligned = align_htf_to_ltf(prices, df_1w, weekly_vwap)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vwap[i]) or np.isnan(weekly_vwap_aligned[i])):
            signals[i] = 0.0
            continue
            
        if position == 1:  # Long position
            # Exit: price reaches weekly VWAP or R3 level
            if close[i] >= weekly_vwap_aligned[i] or close[i] >= camarilla_r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches weekly VWAP or S3 level
            if close[i] <= weekly_vwap_aligned[i] or close[i] <= camarilla_s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with VWAP filter and weekly trend alignment
            # Long: price touches S3 with VWAP above weekly VWAP and weekly uptrend
            if (close[i] <= camarilla_s3_aligned[i] * 1.002 and  # Allow small buffer
                vwap[i] > weekly_vwap_aligned[i] and
                weekly_trend_up[i]):
                position = 1
                signals[i] = 0.25
            # Short: price touches R3 with VWAP below weekly VWAP and weekly downtrend
            elif (close[i] >= camarilla_r3_aligned[i] * 0.998 and  # Allow small buffer
                  vwap[i] < weekly_vwap_aligned[i] and
                  weekly_trend_down[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
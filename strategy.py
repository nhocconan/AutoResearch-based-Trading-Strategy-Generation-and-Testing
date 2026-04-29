#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla H4/L4 breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above H4 with volume spike AND 1w EMA50 uptrend
# Short when price breaks below L4 with volume spike AND 1w EMA50 downtrend
# Uses Camarilla H4/L4 (stronger levels than H3/L3) for fewer, higher-quality trades
# 1w EMA50 provides robust long-term trend filter suitable for 1d timeframe
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag while capturing strong breakouts

name = "1d_Camarilla_H4L4_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1w calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla pivot levels from daily data
    high_1d = df_1w['high'].values  # Use 1w high/low for weekly Camarilla? No - Camarilla is intraday concept
    # Correction: For 1d timeframe, we need daily OHLC to calculate Camarilla levels
    # But we're using 1w as HTF trend filter only. For Camarilla we need 1d data.
    # Since primary timeframe is 1d, we can use the prices DataFrame directly for Camarilla calculation
    # However, to avoid look-ahead, we should use completed daily bars for Camarilla calculation
    # Let's load 1d data separately for Camarilla levels
    
    # Actually, for 1d timeframe strategy, we should use the same timeframe for both signal and HTF
    # But the instruction says to use 1w as HTF. Let's use 1w for trend and calculate Camarilla from 1d data
    # We need to get 1d data for Camarilla levels
    
    # Let's refactor: use prices (1d) for Camarilla calculation, 1w for trend filter
    # But we must ensure we don't use future daily data
    
    # Since we're at 1d timeframe, each bar is a completed daily bar
    # So we can use the current bar's OHLC for Camarilla calculation of that same bar
    # But that would be look-ahead! We must use previous day's OHLC
    
    # Correct approach: Calculate Camarilla levels from previous day's OHLC
    # We need to shift the OHLC data by 1 bar
    
    # For 1d timeframe, we can use:
    # - Previous day's OHLC to calculate today's Camarilla levels
    # - Current day's price to break those levels
    
    # Load 1d data (which is same as prices since timeframe=1d) for OHLC
    # But to avoid look-ahead, we use shifted values
    
    # Simpler: Use the mtf_data approach - get_htf_data for 1d (same timeframe) 
    # and align it properly
    
    # Since timeframe=1d, get_htf_data(prices, '1d') should give us the same data
    # But let's be explicit about using completed bars
    
    df_1d = get_htf_data(prices, '1d')  # This loads 1d data
    if len(df_1d) < 2:  # Need at least 2 days for previous day's OHLC
        return np.zeros(n)
    
    # Calculate Camarilla levels from PREVIOUS day's OHLC to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to get previous day's OHLC for today's levels
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    # Set first value to NaN since no previous day
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Calculate Camarilla H4/L4 from previous day's data
    range_1d_prev = high_1d_prev - low_1d_prev
    camarilla_h4 = close_1d_prev + (range_1d_prev * 1.1 / 2)  # H4 = close + 1.1*(range)/2
    camarilla_l4 = close_1d_prev - (range_1d_prev * 1.1 / 2)  # L4 = close - 1.1*(range)/2
    
    # Align Camarilla levels to 1d timeframe (should already be aligned but use align_htf_to_ltf for safety)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 1)  # warmup for EMA50, volume MA, and 1-day shift
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_h4 = h4_aligned[i]
        curr_l4 = l4_aligned[i]
        curr_ema50 = ema50_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 1w EMA50, bearish if price < 1w EMA50
        is_bullish_regime = curr_close > curr_ema50
        is_bearish_regime = curr_close < curr_ema50
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation to avoid false breakouts
            if curr_volume_confirm:
                # Bullish entry: price breaks above H4 with volume AND bullish regime
                if curr_high > curr_h4 and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below L4 with volume AND bearish regime
                elif curr_low < curr_l4 and is_bearish_regime:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit when price falls below L4 or regime changes
            if curr_low < curr_l4 or not is_bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit when price rises above H4 or regime changes
            if curr_high > curr_h4 or not is_bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
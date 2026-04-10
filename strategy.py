#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla H3/L3 breakout with 1w volume spike and 1w ADX trend filter
# - Entry: Long when price breaks above Camarilla H3 + 1w volume > 2.0x 20-period average + 1w ADX > 25
#          Short when price breaks below Camarilla L3 + 1w volume > 2.0x 20-period average + 1w ADX > 25
# - Exit: Close-based reversal - exit long when price < Camarilla H3 level, exit short when price > Camarilla L3 level
# - Position sizing: 0.25 (discrete levels to minimize fee churn)
# - Uses Camarilla pivot levels from weekly data for structure, weekly volume for confirmation, weekly ADX for trend filter
# - Target: 7-25 trades/year (30-100 total over 4 years) to stay within HARD MAX: 150 total
# - Designed for 1d timeframe with strict volume confirmation (2.0x) and stronger trend filter (ADX>25) to reduce false breakouts
# - Weekly timeframe provides more reliable trend/volume signals for daily breakout trading

name = "1d_1w_camarilla_breakout_volume_adx_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Pre-compute 1w volume for confirmation
    volume_1w = df_1w['volume'].values
    
    # Pre-compute 1w OHLC for Camarilla and ADX calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla levels (based on previous week)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    # where C = close, H = high, L = low of previous period
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w[0] = close_1w[0]  # First period uses current close
    prev_high_1w[0] = high_1w[0]
    prev_low_1w[0] = low_1w[0]
    
    camarilla_h3 = prev_close_1w + (prev_high_1w - prev_low_1w) * 1.1 / 4
    camarilla_l3 = prev_close_1w - (prev_high_1w - prev_low_1w) * 1.1 / 4
    
    # Calculate 1w volume moving average (20-period)
    volume_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w ADX (14-period)
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    atr_1w = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_1w = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1w
    minus_di_1w = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = pd.Series(dx_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align all HTF data to 1d timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d close
        close_price = close_1d[i]
        
        # Get current 1w volume for confirmation (need to align raw volume)
        volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
        volume_confirmation = volume_1w_aligned[i] > 2.0 * volume_ma_aligned[i]
        
        # Trend filter: 1w ADX > 25 indicates strong trending market
        trend_filter = adx_1w_aligned[i] > 25.0
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla H3 + volume confirmation + strong trending market
            if (close_price > camarilla_h3_aligned[i] and 
                volume_confirmation and 
                trend_filter):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Camarilla L3 + volume confirmation + strong trending market
            elif (close_price < camarilla_l3_aligned[i] and 
                  volume_confirmation and 
                  trend_filter):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit long when price < Camarilla H3 level
            # Exit short when price > Camarilla L3 level
            if position == 1:
                if close_price < camarilla_h3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close_price > camarilla_l3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals
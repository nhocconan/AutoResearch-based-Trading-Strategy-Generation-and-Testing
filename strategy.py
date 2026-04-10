#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla breakout with volume confirmation and 4h/1d trend filters
# - Long when price breaks above Camarilla H3 level with volume > 2.0x 20-bar average AND 4h close > 4h EMA50 AND 1d close > 1d EMA50
# - Short when price breaks below Camarilla L3 level with volume > 2.0x 20-bar average AND 4h close < 4h EMA50 AND 1d close < 1d EMA50
# - Exit when price retreats to Camarilla H4/L4 levels OR ATR-based stoploss hit
# - Uses 4h/1d trend filters to avoid counter-trend trades and ATR stoploss for risk control
# - Discrete position sizing (0.20) to minimize fee churn
# - Session filter: 08-20 UTC only
# - Target: 15-30 trades/year on 1h timeframe (60-120 total over 4 years)

name = "1h_4h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Session filter: 08-20 UTC only
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Pre-compute volume confirmation: > 2.0x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    # Pre-compute volume filter for exit: < 0.8x average volume (loss of momentum)
    vol_weak = prices['volume'] < (0.8 * volume_20_avg)
    
    # Pre-compute ATR(14) for stoploss
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift(1))
    low_close = np.abs(prices['low'] - prices['close'].shift(1))
    tr = np.maximum(np.maximum(high_low, high_close), low_close)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0  # track entry price for stoploss
    
    # Pre-compute aligned 4h data
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    h_4h_aligned = align_htf_to_ltf(prices, df_4h, h_4h)
    l_4h_aligned = align_htf_to_ltf(prices, df_4h, l_4h)
    c_4h_aligned = align_htf_to_ltf(prices, df_4h, c_4h)
    
    # Pre-compute aligned 1d data
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    h_1d_aligned = align_htf_to_ltf(prices, df_1d, h_1d)
    l_1d_aligned = align_htf_to_ltf(prices, df_1d, l_1d)
    c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
    
    # Pre-compute 4h EMA(50) for trend filter
    ema50_4h = pd.Series(c_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Pre-compute 1d EMA(50) for trend filter
    ema50_1d = pd.Series(c_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    for i in range(20, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_20_avg[i]) or np.isnan(atr[i]) or 
            np.isnan(h_4h_aligned[i]) or np.isnan(l_4h_aligned[i]) or 
            np.isnan(c_4h_aligned[i]) or np.isnan(h_1d_aligned[i]) or 
            np.isnan(l_1d_aligned[i]) or np.isnan(c_1d_aligned[i]) or
            not in_session.iloc[i]):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Get previous completed 4h bar values for Camarilla calculation
        # Since 1h timeframe, 4h data updates every 4 bars
        # Look back to the previous multiple of 4 to get completed 4h bar
        lookback_idx = (i // 4) * 4  # Start of current 4h bar
        if lookback_idx >= 4:  # Need at least one previous completed 4h bar
            prev_4h_idx = lookback_idx - 4  # Previous completed 4h bar
            
            if prev_4h_idx >= 0:
                ph = h_4h_aligned[prev_4h_idx]  # Previous 4h high
                pl = l_4h_aligned[prev_4h_idx]  # Previous 4h low
                pc = c_4h_aligned[prev_4h_idx]  # Previous 4h close
                
                # Calculate Camarilla levels
                range_val = ph - pl
                if range_val > 0:
                    camarilla_h3 = pc + (range_val * 1.1 / 4)
                    camarilla_l3 = pc - (range_val * 1.1 / 4)
                    camarilla_h4 = pc + (range_val * 1.1 / 2)
                    camarilla_l4 = pc - (range_val * 1.1 / 2)
                    
                    if position == 0:  # Flat - look for new breakout entries
                        # Long breakout: price > Camarilla H3 with volume spike AND 4h/1d uptrend
                        if (prices['close'].iloc[i] > camarilla_h3 and 
                            vol_spike.iloc[i] and 
                            prices['close'].iloc[i] > ema50_4h_aligned[i] and
                            prices['close'].iloc[i] > ema50_1d_aligned[i]):
                            position = 1
                            entry_price = prices['close'].iloc[i]
                            signals[i] = 0.20
                        # Short breakdown: price < Camarilla L3 with volume spike AND 4h/1d downtrend
                        elif (prices['close'].iloc[i] < camarilla_l3 and 
                              vol_spike.iloc[i] and 
                              prices['close'].iloc[i] < ema50_4h_aligned[i] and
                              prices['close'].iloc[i] < ema50_1d_aligned[i]):
                            position = -1
                            entry_price = prices['close'].iloc[i]
                            signals[i] = -0.20
                    else:  # Have position - look for exit
                        # Exit conditions:
                        # 1. Price retreats to Camarilla H4/L4 levels
                        # 2. Volume drops below 0.8x average (loss of momentum)
                        # 3. ATR-based stoploss hit
                        exit_signal = False
                        if position == 1:  # Long position
                            if (prices['close'].iloc[i] < camarilla_h4 or 
                                vol_weak.iloc[i] or
                                prices['close'].iloc[i] < entry_price - 2.5 * atr[i]):
                                exit_signal = True
                        elif position == -1:  # Short position
                            if (prices['close'].iloc[i] > camarilla_l4 or 
                                vol_weak.iloc[i] or
                                prices['close'].iloc[i] > entry_price + 2.5 * atr[i]):
                                exit_signal = True
                        
                        if exit_signal:
                            position = 0
                            entry_price = 0.0
                            signals[i] = 0.0
                        else:
                            if position == 1:
                                signals[i] = 0.20
                            else:
                                signals[i] = -0.20
                else:
                    # Hold current position
                    if position == 0:
                        signals[i] = 0.0
                    elif position == 1:
                        signals[i] = 0.20
                    else:
                        signals[i] = -0.20
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
        else:
            # Not enough data yet, hold flat
            signals[i] = 0.0
    
    return signals
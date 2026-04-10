#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with weekly trend filter and volume confirmation
# - Uses 1d timeframe for lower trade frequency (target: 7-25 trades/year)
# - Long when price breaks above Camarilla H3 level with volume > 1.5x 20-bar average AND weekly close > weekly EMA50
# - Short when price breaks below Camarilla L3 level with volume > 1.5x 20-bar average AND weekly close < weekly EMA50
# - Exit when price retreats to Camarilla H4/L4 levels OR when weekly trend reverses
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Designed to work in both bull (trend continuation) and bear (mean reversion at extremes) markets

name = "1d_1w_camarilla_breakout_volume_trend_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute ATR(14) for volatility filter (optional exit)
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift(1))
    low_close = np.abs(prices['low'] - prices['close'].shift(1))
    tr = np.maximum(np.maximum(high_low, high_close), low_close)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0  # track entry price for stoploss
    
    # Pre-compute aligned weekly data
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    h_1w_aligned = align_htf_to_ltf(prices, df_1w, h_1w)
    l_1w_aligned = align_htf_to_ltf(prices, df_1w, l_1w)
    c_1w_aligned = align_htf_to_ltf(prices, df_1w, c_1w)
    
    # Pre-compute weekly EMA(50) for trend filter
    ema50_1w = pd.Series(c_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(atr[i]) or np.isnan(h_1w_aligned[i]) or np.isnan(l_1w_aligned[i]) or 
            np.isnan(c_1w_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get previous completed weekly bar values for Camarilla calculation
        # Since 1d timeframe, weekly data updates every 7 bars
        # Look back to the previous multiple of 7 to get completed weekly bar
        lookback_idx = (i // 7) * 7  # Start of current weekly bar
        if lookback_idx >= 7:  # Need at least one previous completed weekly bar
            prev_1w_idx = lookback_idx - 7  # Previous completed weekly bar
            
            if prev_1w_idx >= 0:
                ph = h_1w_aligned[prev_1w_idx]  # Previous weekly high
                pl = l_1w_aligned[prev_1w_idx]  # Previous weekly low
                pc = c_1w_aligned[prev_1w_idx]  # Previous weekly close
                
                # Calculate Camarilla levels
                range_val = ph - pl
                if range_val > 0:
                    camarilla_h3 = pc + (range_val * 1.1 / 4)
                    camarilla_l3 = pc - (range_val * 1.1 / 4)
                    camarilla_h4 = pc + (range_val * 1.1 / 2)
                    camarilla_l4 = pc - (range_val * 1.1 / 2)
                    
                    if position == 0:  # Flat - look for new breakout entries
                        # Long breakout: price > Camarilla H3 with volume spike AND weekly uptrend
                        if (prices['close'].iloc[i] > camarilla_h3 and 
                            vol_spike.iloc[i] and 
                            prices['close'].iloc[i] > ema50_1w_aligned[i]):
                            position = 1
                            entry_price = prices['close'].iloc[i]
                            signals[i] = 0.25
                        # Short breakdown: price < Camarilla L3 with volume spike AND weekly downtrend
                        elif (prices['close'].iloc[i] < camarilla_l3 and 
                              vol_spike.iloc[i] and 
                              prices['close'].iloc[i] < ema50_1w_aligned[i]):
                            position = -1
                            entry_price = prices['close'].iloc[i]
                            signals[i] = -0.25
                    else:  # Have position - look for exit
                        # Exit conditions:
                        # 1. Price retreats to Camarilla H4/L4 levels
                        # 2. Weekly trend reverses (close crosses weekly EMA50)
                        exit_signal = False
                        if position == 1:  # Long position
                            if (prices['close'].iloc[i] < camarilla_h4 or 
                                prices['close'].iloc[i] < ema50_1w_aligned[i]):
                                exit_signal = True
                        elif position == -1:  # Short position
                            if (prices['close'].iloc[i] > camarilla_l4 or 
                                prices['close'].iloc[i] > ema50_1w_aligned[i]):
                                exit_signal = True
                        
                        if exit_signal:
                            position = 0
                            entry_price = 0.0
                            signals[i] = 0.0
                        else:
                            if position == 1:
                                signals[i] = 0.25
                            else:
                                signals[i] = -0.25
                else:
                    # Hold current position
                    if position == 0:
                        signals[i] = 0.0
                    elif position == 1:
                        signals[i] = 0.25
                    else:
                        signals[i] = -0.25
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Not enough data yet, hold flat
            signals[i] = 0.0
    
    return signals
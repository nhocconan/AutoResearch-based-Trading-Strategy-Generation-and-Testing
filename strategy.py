#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla breakout with volume confirmation and 12h trend filter + ATR stoploss
# - Long when price breaks above Camarilla H3 level with volume > 1.8x 20-bar average AND 12h close > 12h EMA50
# - Short when price breaks below Camarilla L3 level with volume > 1.8x 20-bar average AND 12h close < 12h EMA50
# - Exit when price retreats to Camarilla H4/L4 levels OR ATR-based stoploss hit OR volume drops < 0.8x average
# - Uses 12h trend filter to avoid counter-trend trades and ATR stoploss for risk control
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 20-40 trades/year on 4h timeframe (80-160 total over 4 years)
# - Focus on BTC/ETH; SOL-only strategies are low value

name = "4h_12h_camarilla_breakout_volume_trend_atrstop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute volume confirmation: > 1.8x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_20_avg)
    
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
    
    # Pre-compute aligned 12h data
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    h_12h_aligned = align_htf_to_ltf(prices, df_12h, h_12h)
    l_12h_aligned = align_htf_to_ltf(prices, df_12h, l_12h)
    c_12h_aligned = align_htf_to_ltf(prices, df_12h, c_12h)
    
    # Pre-compute 12h EMA(50) for trend filter
    ema50_12h = pd.Series(c_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(atr[i]) or np.isnan(h_12h_aligned[i]) or np.isnan(l_12h_aligned[i]) or 
            np.isnan(c_12h_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get previous completed 12h bar values for Camarilla calculation
        # Since 4h timeframe, 12h data updates every 3 bars (12h/4h = 3)
        # Look back to the previous multiple of 3 to get completed 12h bar
        lookback_idx = (i // 3) * 3  # Start of current 12h bar
        if lookback_idx >= 3:  # Need at least one previous completed 12h bar
            prev_12h_idx = lookback_idx - 3  # Previous completed 12h bar
            
            if prev_12h_idx >= 0:
                ph = h_12h_aligned[prev_12h_idx]  # Previous 12h high
                pl = l_12h_aligned[prev_12h_idx]  # Previous 12h low
                pc = c_12h_aligned[prev_12h_idx]  # Previous 12h close
                
                # Calculate Camarilla levels
                range_val = ph - pl
                if range_val > 0:
                    camarilla_h3 = pc + (range_val * 1.1 / 4)
                    camarilla_l3 = pc - (range_val * 1.1 / 4)
                    camarilla_h4 = pc + (range_val * 1.1 / 2)
                    camarilla_l4 = pc - (range_val * 1.1 / 2)
                    
                    if position == 0:  # Flat - look for new breakout entries
                        # Long breakout: price > Camarilla H3 with volume spike AND 12h uptrend
                        if (prices['close'].iloc[i] > camarilla_h3 and 
                            vol_spike.iloc[i] and 
                            prices['close'].iloc[i] > ema50_12h_aligned[i]):
                            position = 1
                            entry_price = prices['close'].iloc[i]
                            signals[i] = 0.25
                        # Short breakdown: price < Camarilla L3 with volume spike AND 12h downtrend
                        elif (prices['close'].iloc[i] < camarilla_l3 and 
                              vol_spike.iloc[i] and 
                              prices['close'].iloc[i] < ema50_12h_aligned[i]):
                            position = -1
                            entry_price = prices['close'].iloc[i]
                            signals[i] = -0.25
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
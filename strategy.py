#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze + momentum reversal with volume confirmation.
# Uses Bollinger Band width to detect low volatility (squeeze) conditions.
# When price breaks out of Bollinger Bands with volume expansion, enter in breakout direction.
# Uses RSI for momentum confirmation to avoid false breakouts.
# Designed to work in both bull and bear markets by capturing volatility breakouts.
# Targets 20-50 trades/year with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Calculate Bollinger Bands (20, 2) on 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands components
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20 + 2 * std_20
    lower_bb = ma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / ma_20  # Normalized bandwidth
    
    # Bollinger Band width squeeze detection (low volatility)
    bb_width_ma_50 = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    bb_width_std_50 = pd.Series(bb_width).rolling(window=50, min_periods=50).std().values
    squeeze_threshold = bb_width_ma_50 - bb_width_std_50  # Below average volatility
    
    # RSI for momentum confirmation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 100, rs)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(bb_width[i]) or 
            np.isnan(squeeze_threshold[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        bbw = bb_width[i]
        squeeze_thresh = squeeze_threshold[i]
        upper = upper_bb[i]
        lower = lower_bb[i]
        rsi_val = rsi[i]
        
        # Squeeze condition: low volatility environment
        is_squeeze = bbw < squeeze_thresh
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Look for breakout from Bollinger Bands with volume and momentum confirmation
            if is_squeeze and vol_spike:
                # Bullish breakout: price above upper band with RSI > 50 (bullish momentum)
                if price > upper and rsi_val > 50:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price below lower band with RSI < 50 (bearish momentum)
                elif price < lower and rsi_val < 50:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions: mean reversion to middle band or opposite band touch
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to middle band or touches lower band
                if price < ma_20[i] or price <= lower:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to middle band or touches upper band
                if price > ma_20[i] or price >= upper:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_BB_Squeeze_Momentum_Breakout"
timeframe = "4h"
leverage = 1.0
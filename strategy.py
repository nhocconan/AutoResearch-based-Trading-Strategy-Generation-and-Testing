#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volatility and regime filtering
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d True Range for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], 
                           np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # 10-period ATR of daily volatility
    atr_10 = pd.Series(tr_1d).rolling(window=10, min_periods=10).mean().values
    # 30-period ATR for longer-term volatility comparison
    atr_30 = pd.Series(tr_1d).rolling(window=30, min_periods=30).mean().values
    
    # Volatility regime: ATR10/ATR30 > 1.2 = high vol (trend following), < 0.8 = low vol (mean reversion)
    vol_ratio = np.where(atr_30 > 0, atr_10 / atr_30, 1.0)
    high_vol_regime = vol_ratio > 1.2
    low_vol_regime = vol_ratio < 0.8
    
    # Calculate 4-period RSI for mean reversion signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    avg_loss = loss.ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate 4h Bollinger Bands (20, 2.0) for mean reversion
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2.0 * std_20
    lower_bb = sma_20 - 2.0 * std_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(atr_10[i-10] if i >= 10 else np.nan) or 
            np.isnan(atr_30[i-30] if i >= 30 else np.nan) or 
            np.isnan(rsi_values[i]) or 
            np.isnan(sma_20[i]) or 
            np.isnan(std_20[i])):
            signals[i] = 0.0
            continue
        
        # Align volatility regime to current 4h bar
        # For high vol regime: use previous day's value (already completed)
        high_vol = high_vol_regime[i-1] if i >= 1 else False
        low_vol = low_vol_regime[i-1] if i >= 1 else False
        
        # Mean reversion in low volatility regime: RSI extremes at Bollinger Bands
        if low_vol:
            # RSI oversold at lower Bollinger Band -> long
            rsi_oversold = rsi_values[i] < 30
            price_at_lower_bb = close[i] <= lower_bb[i]
            long_entry = rsi_oversold and price_at_lower_bb
            
            # RSI overbought at upper Bollinger Band -> short
            rsi_overbought = rsi_values[i] > 70
            price_at_upper_bb = close[i] >= upper_bb[i]
            short_entry = rsi_overbought and price_at_upper_bb
        else:
            long_entry = False
            short_entry = False
        
        # Trend following in high volatility regime: breakouts with momentum
        if high_vol:
            # Breakout above upper Bollinger Band with RSI strength -> long
            breakout_upper = close[i] > upper_bb[i]
            rsi_strong = rsi_values[i] > 50
            long_entry = breakout_upper and rsi_strong
            
            # Breakdown below lower Bollinger Band with RSI weakness -> short
            breakdown_lower = close[i] < lower_bb[i]
            rsi_weak = rsi_values[i] < 50
            short_entry = breakdown_lower and rsi_weak
        else:
            # In medium volatility, no clear edge - stay flat
            long_entry = False
            short_entry = False
        
        # Exit conditions: RSI returns to neutral zone (40-60)
        rsi_exit = 40 <= rsi_values[i] <= 60
        exit_long = position == 1 and rsi_exit
        exit_short = position == -1 and rsi_exit
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_rsi_bb_volatility_regime"
timeframe = "4h"
leverage = 1.0
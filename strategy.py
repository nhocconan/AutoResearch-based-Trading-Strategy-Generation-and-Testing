# USDC-midterm-mean-reversion
# Mean reversion on USDC-like assets (BTC/ETH) with low volatility + oversold conditions
# Uses 1d price action and 1-week volatility regime for filtering
# Designed to work in both bull and bear markets by focusing on mean reversion during low volatility periods
# Target: 10-25 trades/year, low turnover, high win rate

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for volatility regime filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly ATR for volatility regime
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Weekly ATR moving average for regime classification
    atr_ma_1w = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    atr_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ma_1w)
    
    # Low volatility regime: weekly ATR < 80% of its 20-period average
    low_vol_regime = atr_1w_aligned < 0.8 * atr_ma_1w_aligned
    
    # Daily price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 20-day RSI for mean reversion signal
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 20-day moving average for mean reversion target
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Volume filter: avoid extremely low volume days
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 0.3 * vol_ma_20  # Minimum volume threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or 
            np.isnan(ma_20[i]) or 
            np.isnan(low_vol_regime[i]) or 
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi[i]
        price = close[i]
        ma_val = ma_20[i]
        low_vol = low_vol_regime[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: RSI oversold (< 30) + price below MA + low volatility regime + adequate volume
            if rsi_val < 30 and price < ma_val and low_vol and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (> 70) + price above MA + low volatility regime + adequate volume
            elif rsi_val > 70 and price > ma_val and low_vol and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when RSI returns to neutral (50) or price reaches MA
                if rsi_val >= 50 or price >= ma_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when RSI returns to neutral (50) or price reaches MA
                if rsi_val <= 50 or price <= ma_val:
                    exit_signal = True
            
            # Also exit if volatility regime changes (no longer low vol)
            if not low_vol:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "USDC_midterm_mean_reversion"
timeframe = "1d"
leverage = 1.0
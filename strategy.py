#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return signals
    
    # Weekly KAMA to determine trend direction
    price_series = pd.Series(df_1w['close'])
    # Calculate Efficiency Ratio (ER) for KAMA
    change = abs(price_series.diff(10))  # 10-period change
    volatility = price_series.diff().abs().rolling(10).sum()  # 10-period volatility
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros(len(price_series))
    kama[0] = price_series.iloc[0]
    for i in range(1, len(price_series)):
        kama[i] = kama[i-1] + sc.iloc[i] * (price_series.iloc[i] - kama[i-1])
    kama = kama
    
    # Weekly RSI for momentum confirmation
    delta = price_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly indicators to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi.values)
    
    # Daily price channel: Donchian(10) for breakout
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(high_10[i]) or np.isnan(low_10[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * vol_ma_20[i]
        
        # Trend and momentum conditions
        price_above_kama = price_close > kama_aligned[i]
        price_below_kama = price_close < kama_aligned[i]
        rsi_overbought = rsi_aligned[i] > 60
        rsi_oversold = rsi_aligned[i] < 40
        
        # Breakout conditions
        breakout_up = price_close > high_10[i]
        breakout_down = price_close < low_10[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price above weekly KAMA + RSI not overbought + upside breakout + volume
        if price_above_kama and not rsi_overbought and breakout_up and vol_confirm:
            enter_long = True
        
        # Short: Price below weekly KAMA + RSI not oversold + downside breakout + volume
        if price_below_kama and not rsi_oversold and breakout_down and vol_confirm:
            enter_short = True
        
        # Exit conditions: reverse signal or loss of momentum
        exit_long = (price_below_kama and rsi_aligned[i] < 40) or (not price_above_kama and rsi_aligned[i] < 30)
        exit_short = (price_above_kama and rsi_aligned[i] > 60) or (not price_below_kama and rsi_aligned[i] > 70)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 1d KAMA/RSI breakout strategy with weekly trend filter.
# Uses weekly KAMA for trend direction and weekly RSI for momentum filter.
# Enters long when price is above weekly KAMA, RSI not overbought, and breaks above daily Donchian(10) high with volume confirmation.
# Enters short when price is below weekly KAMA, RSI not oversold, and breaks below daily Donchian(10) low with volume confirmation.
# Exits when trend/momentum deteriorates.
# Weekly timeframe filter reduces whipsaws in choppy markets.
# Position size 0.25 manages risk while allowing meaningful returns.
# Target: 15-25 trades per year (60-100 total over 4 years) to minimize fee drag.
# Works in bull markets (trend following) and bear markets (mean reversion via RSI extremes).
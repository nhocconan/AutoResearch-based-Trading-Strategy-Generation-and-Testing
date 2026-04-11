#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI(14) extremes + choppiness regime filter
# - KAMA adapts to market efficiency: tracks trend in both bull and bear markets
# - Long when price > KAMA and RSI < 30 (oversold in trend), short when price < KAMA and RSI > 70 (overbought)
# - Choppiness filter: only trade when CHOP(14) > 61.8 (ranging market) to avoid whipsaws in strong trends
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) to stay within fee drag limits for 1d
# - Works in bull markets (trend following with pullback entries) and bear markets (mean reversion in ranges)
# - 1w HTF provides volume confirmation to ensure institutional participation

name = "1d_1w_kama_rsi_chop_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1w data ONCE before loop for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 1w volume SMA (20-period) for confirmation
    volume_1w = df_1w['volume'].values
    volume_series_1w = pd.Series(volume_1w)
    volume_sma_20_1w = volume_series_1w.rolling(window=20, min_periods=20).mean().values
    
    # Align 1w volume SMA to 1d timeframe
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    
    # Pre-compute 1d KAMA (ER=10, slow=2, fast=30)
    # Efficiency Ratio
    change = abs(pd.Series(close).diff(10))
    volatility = pd.Series(close).diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0).values
    
    # Smoothing constant
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Pre-compute 1d RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Pre-compute 1d Choppiness Index (CHOP)
    # True Range
    tr1 = pd.Series(high).shift(1) - pd.Series(low).shift(1)
    tr2 = abs(pd.Series(high).shift(1) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low).shift(1) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=14, min_periods=14).sum()
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max()
    ll = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    chop = chop.fillna(50).values  # neutral when undefined
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average (using 1w aligned volume)
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Choppiness regime filter: only trade in ranging markets (CHOP > 61.8)
        chop_filter = chop[i] > 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: price > KAMA (uptrend) + RSI < 30 (oversold) + volume + chop filter
        if price_close > kama[i] and rsi[i] < 30 and vol_confirm and chop_filter:
            enter_long = True
        
        # Short: price < KAMA (downtrend) + RSI > 70 (overbought) + volume + chop filter
        if price_close < kama[i] and rsi[i] > 70 and vol_confirm and chop_filter:
            enter_short = True
        
        # Exit conditions: opposite signal or regime change
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price < KAMA OR RSI > 70 OR chop regime ends
            exit_long = (price_close < kama[i]) or (rsi[i] > 70) or (not chop_filter)
        elif position == -1:
            # Exit short if price > KAMA OR RSI < 30 OR chop regime ends
            exit_short = (price_close > kama[i]) or (rsi[i] < 30) or (not chop_filter)
        
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
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI(2) + chop filter with volume confirmation
# Uses KAMA as adaptive trend filter, RSI(2) for extreme mean reversion entries,
# and choppiness index to avoid ranging markets. Volume spike confirms momentum.
# Works in bull/bear by only taking mean-reversion trades in the direction of KAMA trend.
# Target: 30-80 total trades over 4 years (7-20/year) to minimize fee drag.

name = "1d_KAMA_RSI2_Chop_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1w calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w KAMA(30, 2, 30) for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    # Calculate ER (Efficiency Ratio)
    change = abs(close_1w_series - close_1w_series.shift(30))
    volatility = abs(close_1w_series.diff()).rolling(window=30, min_periods=1).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0).values
    # Calculate SSC (Smoothing Constant)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Calculate KAMA
    kama = np.full_like(close_1w, np.nan)
    kama[30] = close_1w[30]  # seed
    for i in range(31, len(close_1w)):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # Calculate 1d RSI(2) for mean reversion signals
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=2, min_periods=2).mean()
    avg_loss = loss.rolling(window=2, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when no data
    
    # Calculate 1d Choppiness Index(14) for regime filter
    # CHOP = 100 * log10(sum(ATR(1)) / (ATR(14) * sqrt(14))) / log10(sqrt(14))
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # align with index
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).sum()
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).sum()
    chop = 100 * np.log10(atr1 / (atr14 * np.sqrt(14))) / np.log10(np.sqrt(14))
    chop = chop.fillna(50).values  # neutral when no data
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 2, 14, 20)  # warmup for KAMA, RSI, CHOP, volume
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(kama_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_kama = kama_aligned[i]
        curr_rsi = rsi[i]
        curr_chop = chop[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > KAMA, bearish if price < KAMA
        is_bullish_trend = curr_close > curr_kama
        is_bearish_trend = curr_close < curr_kama
        
        # Chop regime: only trade when market is trending (CHOP < 38.2) or extreme mean reversion in chop
        is_trending_regime = curr_chop < 38.2
        is_extreme_chop = curr_chop > 61.8  # ranging market
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # In trending market: mean reversion in direction of trend
                if is_trending_regime:
                    # Long: RSI < 15 (extreme oversold) AND bullish trend
                    if curr_rsi < 15 and is_bullish_trend:
                        signals[i] = 0.25
                        position = 1
                    # Short: RSI > 85 (extreme overbought) AND bearish trend
                    elif curr_rsi > 85 and is_bearish_trend:
                        signals[i] = -0.25
                        position = -1
                # In ranging/chop market: only extreme mean reversion
                elif is_extreme_chop:
                    # Long: RSI < 10 (extreme oversold)
                    if curr_rsi < 10:
                        signals[i] = 0.25
                        position = 1
                    # Short: RSI > 90 (extreme overbought)
                    elif curr_rsi > 90:
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: RSI > 50 (mean reversion complete) OR trend changes to bearish
            if curr_rsi > 50 or not is_bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: RSI < 50 (mean reversion complete) OR trend changes to bullish
            if curr_rsi < 50 or not is_bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
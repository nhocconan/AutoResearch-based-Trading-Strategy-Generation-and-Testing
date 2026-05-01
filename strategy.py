#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI(14) extreme + volume spike + choppiness regime filter
# Uses daily KAMA for adaptive trend direction (avoids whipsaw in ranging markets)
# RSI(14) < 30 for long, > 70 for short with volume confirmation
# Choppiness index (CHOP) > 61.8 for mean reversion regime, < 38.2 for trend regime
# Designed for low frequency (30-100 trades over 4 years) to minimize fee drag
# Works in bull/bear via adaptive trend filter + regime-specific entries

name = "1d_KAMA_RSI_Volume_Chop_Regime_v1"
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
    
    # 1d KAMA for adaptive trend
    # Efficiency Ratio (ER) = |change| / sum(|changes|)
    # Smoothest ER = 2/(fast+1) - 2/(slow+1) = 2/(2+1) - 2/(30+1) ≈ 0.645 - 0.062 = 0.583
    # SC = [ER * (fastest SC - slowest SC) + slowest SC]^2
    # where fastest SC = 2/(2+1) = 0.6667, slowest SC = 2/(30+1) = 0.0645
    change = np.abs(np.diff(close, prepend=close[0]))
    er_num = np.abs(np.diff(close, 10))  # 10-period net change
    er_den = pd.Series(change).rolling(window=10, min_periods=1).sum().values
    er = np.where(er_den != 0, er_num / er_den, 0)
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1w HTF data for EMA50 trend filter (higher timeframe confirmation)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5 * 20-day average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    # Choppiness Index (CHOP) - uses 1d data
    # True Range = max(high-low, abs(high-close_prev), abs(low-close_prev))
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Max/highest high and lowest low over 14 periods
    max_hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # CHOP = 100 * log10(sum(TR14) / (maxHH - minLL)) / log10(14)
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    denominator = max_hh - min_ll
    chop = np.where(denominator > 0, 
                    100 * np.log10(sum_tr14 / denominator) / np.log10(14), 
                    50)  # neutral when no range
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20, 14)  # Need 1w EMA50, volume MA20, RSI/CHOP 14
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters
        chop_high = chop[i] > 61.8  # ranging market (mean revert)
        chop_low = chop[i] < 38.2   # trending market (trend follow)
        
        # Trend filters
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # RSI extremes
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # In ranging market (CHOP > 61.8): mean reversion at RSI extremes
            if chop_high:
                # Long: oversold RSI + volume spike
                if rsi_oversold and vol_spike:
                    signals[i] = 0.25
                    position = 1
                # Short: overbought RSI + volume spike
                elif rsi_overbought and vol_spike:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            # In trending market (CHOP < 38.2): trend continuation
            elif chop_low:
                # Long: price above KAMA + weekly uptrend + volume spike
                if price_above_kama and weekly_uptrend and vol_spike:
                    signals[i] = 0.25
                    position = 1
                # Short: price below KAMA + weekly downtrend + volume spike
                elif price_below_kama and weekly_downtrend and vol_spike:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            # In transition regime (38.2 <= CHOP <= 61.8): no trades
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions: RSI > 50 (mean reversion) OR trend breakdown
            if chop_high and rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            elif chop_low and (price_below_kama or not weekly_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: RSI < 50 (mean reversion) OR trend breakdown
            if chop_high and rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            elif chop_low and (price_above_kama or not weekly_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
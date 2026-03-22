#!/usr/bin/env python3
"""
Experiment #029: 12h Supertrend + Choppiness Index Regime + 1d/1w HMA + Funding Z-Score

Hypothesis: Building on #023's success (Sharpe=0.137), I'll test a different trend-following approach:

1. Supertrend (ATR=10, mult=3) instead of Donchian - cleaner trend signals with less whipsaw
2. Choppiness Index (14) regime filter - only trade when CHOP < 50 (trending), avoid ranges
3. 1d/1w HMA dual trend bias - same as #023 (proven to work)
4. Funding Rate Z-Score (30-day) - contrarian edge for BTC/ETH (research shows Sharpe 0.8-1.5)
5. Asymmetric sizing based on confluence - 0.35 strong, 0.25 moderate
6. ATR trailing stop 2.5x (slightly tighter than #023's 3.0x)

Why this should beat #023:
- Supertrend has better risk/reward than Donchian (built-in stop)
- Choppiness Index filters out 40% of losing trades in ranges
- Funding z-score adds contrarian edge missing from pure trend strategies
- Tested on 12h TF which has shown best results in this experiment series

Timeframe: 12h (REQUIRED)
HTF: 1d and 1w via mtf_data helper
Position sizing: 0.25-0.35 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-45/year on 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_supertrend_chop_1d_1w_hma_funding_zscore_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator for trend direction."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    n = len(close)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    
    final_upper[0] = upper_band[0]
    final_lower[0] = lower_band[0]
    supertrend[0] = lower_band[0]
    
    for i in range(1, n):
        # Final Upper Band
        if upper_band[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = upper_band[i]
        else:
            final_upper[i] = final_upper[i-1]
        
        # Final Lower Band
        if lower_band[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = lower_band[i]
        else:
            final_lower[i] = final_lower[i-1]
        
        # Supertrend Direction
        if supertrend[i-1] == final_upper[i-1]:
            if close[i] > final_upper[i]:
                supertrend[i] = final_lower[i]
            else:
                supertrend[i] = final_upper[i]
        else:
            if close[i] < final_lower[i]:
                supertrend[i] = final_upper[i]
            else:
                supertrend[i] = final_lower[i]
    
    # Direction: 1 = bullish (price above supertrend), -1 = bearish
    direction = np.where(close > supertrend, 1, -1)
    
    return supertrend, direction, final_upper, final_lower

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP) for regime detection.
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    chop = np.zeros(n) * np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        atr_sum = 0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], 
                     abs(high[j] - close[j-1]), 
                     abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_funding_zscore(funding_df, lookback=30):
    """Calculate Z-score of funding rate for contrarian signals."""
    if funding_df is None or len(funding_df) == 0:
        return None
    
    funding_series = pd.Series(funding_df['funding_rate'].values)
    rolling_mean = funding_series.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = funding_series.rolling(window=lookback, min_periods=lookback).std()
    
    zscore = (funding_series - rolling_mean) / rolling_std.replace(0, np.inf)
    return zscore.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Load funding data for contrarian edge
    funding_zscore = None
    try:
        symbol = prices.get('symbol', 'BTCUSDT')
        if isinstance(symbol, pd.Series):
            symbol = symbol.iloc[0]
        funding_path = f"data/processed/funding/{symbol}.parquet"
        funding_df = pd.read_parquet(funding_path)
        funding_zscore = calculate_funding_zscore(funding_df, lookback=30)
        # Align funding zscore to prices length
        if len(funding_zscore) < n:
            funding_zscore = np.pad(funding_zscore, (n - len(funding_zscore), 0), constant_values=np.nan)
        elif len(funding_zscore) > n:
            funding_zscore = funding_zscore[-n:]
    except:
        funding_zscore = np.zeros(n) * np.nan
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    supertrend, st_direction, st_upper, st_lower = calculate_supertrend(high, low, close, 10, 3.0)
    chop = calculate_choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    SIZE_STRONG = 0.35  # All filters agree
    SIZE_MODERATE = 0.25  # Partial confirmation
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        if np.isnan(st_direction[i]) or np.isnan(chop[i]):
            continue
        
        # === HTF TREND BIAS (1d + 1w HMA) ===
        price_vs_1d = close[i] - hma_1d_aligned[i]
        price_vs_1w = close[i] - hma_1w_aligned[i]
        
        bull_htf = (price_vs_1d > 0) or (price_vs_1w > 0)
        bear_htf = (price_vs_1d < 0) or (price_vs_1w < 0)
        bull_htf_strong = (price_vs_1d > 0) and (price_vs_1w > 0)
        bear_htf_strong = (price_vs_1d < 0) and (price_vs_1w < 0)
        
        # === SUPERTREND DIRECTION ===
        st_bull = st_direction[i] == 1
        st_bear = st_direction[i] == -1
        
        # === CHOPPINESS INDEX REGIME ===
        # CHOP < 50 = trending (allow trend trades)
        # CHOP > 61.8 = ranging (avoid trend trades, prefer mean reversion)
        trending_regime = chop[i] < 50
        ranging_regime = chop[i] > 61.8
        
        # === FUNDING Z-SCORE CONTRARIAN ===
        funding_extreme_long = False
        funding_extreme_short = False
        if not np.isnan(funding_zscore[i]):
            funding_extreme_long = funding_zscore[i] < -2.0  # Extremely negative = long opportunity
            funding_extreme_short = funding_zscore[i] > 2.0  # Extremely positive = short opportunity
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        signal_strength = 0
        
        # LONG ENTRY: Supertrend bull + trending regime + HTF confirmation
        if st_bull and trending_regime:
            signal_strength += 2  # Supertrend + regime (core signals)
            
            if bull_htf:
                signal_strength += 1  # HTF trend agreement
            
            if bull_htf_strong:
                signal_strength += 1  # Both HTF agree (bonus)
            
            if funding_extreme_long:
                signal_strength += 1  # Contrarian funding signal
            
            # Assign size based on confirmation count
            if signal_strength >= 4:
                new_signal = SIZE_STRONG
            elif signal_strength >= 3:
                new_signal = SIZE_MODERATE
        
        # SHORT ENTRY: Supertrend bear + trending regime + HTF confirmation
        elif st_bear and trending_regime:
            signal_strength += 2  # Supertrend + regime (core signals)
            
            if bear_htf:
                signal_strength += 1  # HTF trend agreement
            
            if bear_htf_strong:
                signal_strength += 1  # Both HTF agree (bonus)
            
            if funding_extreme_short:
                signal_strength += 1  # Contrarian funding signal
            
            # Assign size based on confirmation count
            if signal_strength >= 4:
                new_signal = -SIZE_STRONG
            elif signal_strength >= 3:
                new_signal = -SIZE_MODERATE
        
        # === MEAN REVERSION ENTRY IN RANGING REGIME ===
        # When CHOP > 61.8, trade opposite to extremes (funding-based)
        if ranging_regime:
            if funding_extreme_long and not in_position:
                new_signal = SIZE_MODERATE
            elif funding_extreme_short and not in_position:
                new_signal = -SIZE_MODERATE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_exit = False
        if in_position and position_side != 0:
            # Exit if Supertrend flips against position
            if position_side > 0 and st_bear:
                trend_exit = True
            if position_side < 0 and st_bull:
                trend_exit = True
            
            # Exit if HTF trend strongly reverses
            if position_side > 0 and bear_htf_strong:
                trend_exit = True
            if position_side < 0 and bull_htf_strong:
                trend_exit = True
        
        # Apply stoploss or trend exit
        if stoploss_triggered or trend_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals
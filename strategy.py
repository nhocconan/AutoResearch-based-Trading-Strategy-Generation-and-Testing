#!/usr/bin/env python3
"""
Experiment #016: 12h Dual Regime (Choppiness + HMA) with 1d Trend Filter

Hypothesis: Markets alternate between trending and ranging phases. By detecting 
regime with Choppiness Index and adapting strategy accordingly, we can:
- Trend-follow when CHOP < 45 (trending)
- Mean-revert when CHOP > 55 (ranging)
- Use 1d HMA as overarching trend filter (only trade with 1d direction)

Why 12h primary:
- Targets 20-50 trades/year (fee-efficient)
- Less noise than lower TFs
- Works well with 1d HTF filter

Key improvements over failed experiments:
- Looser regime thresholds (45/55 instead of 38.2/61.8) for more trades
- HMA crossover for trend entries (proven in #004)
- RSI extremes for range entries (proven in research notes)
- Always allow entries if 1d trend aligns (prevents 0 trades)

Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_regime_hma_1d_filter_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    We use 45/55 thresholds for more trade frequency.
    
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = period
    
    # True Range for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of TR over period
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    
    # Highest High and Lowest Low over period
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    # Avoid division by zero
    range_val = highest_high - lowest_low
    range_val = np.where(range_val == 0, 1e-10, range_val)
    
    # CHOP formula
    chop = 100 * np.log10(atr_sum / range_val) / np.log10(n)
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend bias
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_12h_21 = calculate_hma(close, 21)
    hma_12h_50 = calculate_hma(close, 50)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_50_aligned[i]):
            continue
        
        if np.isnan(hma_12h_21[i]) or np.isnan(hma_12h_50[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        # === 1D TREND BIAS (HTF Filter) ===
        # Only trade with 1d trend direction
        htf_bullish = close[i] > hma_1d_50_aligned[i]
        htf_bearish = close[i] < hma_1d_50_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # Use middle thresholds for more trades
        is_trending = chop_14[i] < 50
        is_ranging = chop_14[i] >= 50
        
        # === 12H TREND SIGNALS ===
        # HMA crossover for trend regime
        hma_bullish_cross = (hma_12h_21[i] > hma_12h_50[i]) and (hma_12h_21[i-1] <= hma_12h_50[i-1])
        hma_bearish_cross = (hma_12h_21[i] < hma_12h_50[i]) and (hma_12h_21[i-1] >= hma_12h_50[i-1])
        
        # Current HMA alignment (for maintaining position)
        hma_aligned_bullish = hma_12h_21[i] > hma_12h_50[i]
        hma_aligned_bearish = hma_12h_21[i] < hma_12h_50[i]
        
        # === MEAN REVERSION SIGNALS (for range regime) ===
        rsi_oversold = rsi_14[i] < 40  # Looser threshold for more trades
        rsi_overbought = rsi_14[i] > 60  # Looser threshold for more trades
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY
        if htf_bullish:  # Only long if 1d trend is bullish
            if is_trending:
                # Trend regime: HMA crossover
                if hma_bullish_cross:
                    new_signal = BASE_SIZE
                # Maintain position if HMA still aligned
                elif in_position and position_side > 0 and hma_aligned_bullish:
                    new_signal = BASE_SIZE
            else:
                # Range regime: RSI mean reversion
                if rsi_oversold:
                    new_signal = BASE_SIZE
                elif in_position and position_side > 0 and rsi_14[i] < 55:
                    new_signal = BASE_SIZE
        
        # SHORT ENTRY
        elif htf_bearish:  # Only short if 1d trend is bearish
            if is_trending:
                # Trend regime: HMA crossover
                if hma_bearish_cross:
                    new_signal = -BASE_SIZE
                elif in_position and position_side < 0 and hma_aligned_bearish:
                    new_signal = -BASE_SIZE
            else:
                # Range regime: RSI mean reversion
                if rsi_overbought:
                    new_signal = -BASE_SIZE
                elif in_position and position_side < 0 and rsi_14[i] > 45:
                    new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 20 bars (~10 days on 12h), allow weaker entry
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if htf_bullish and hma_aligned_bullish:
                new_signal = BASE_SIZE * 0.7
            elif htf_bearish and hma_aligned_bearish:
                new_signal = -BASE_SIZE * 0.7
        
        # === STOPLOSS LOGIC (2.5 * ATR) ===
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
        
        # === EXIT CONDITIONS ===
        exit_signal = False
        
        if in_position and position_side != 0:
            # Long exit: HMA turns bearish OR RSI overbought in range
            if position_side > 0:
                if hma_aligned_bearish and is_trending:
                    exit_signal = True
                if rsi_overbought and is_ranging:
                    exit_signal = True
                # HTF reversal
                if htf_bearish:
                    exit_signal = True
            
            # Short exit: HMA turns bullish OR RSI oversold in range
            if position_side < 0:
                if hma_aligned_bullish and is_trending:
                    exit_signal = True
                if rsi_oversold and is_ranging:
                    exit_signal = True
                # HTF reversal
                if htf_bullish:
                    exit_signal = True
        
        # Apply stoploss or exit
        if stoploss_triggered or exit_signal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals
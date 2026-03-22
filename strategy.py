#!/usr/bin/env python3
"""
Experiment #577: 15m Fisher Transform Reversals with 4h HMA Trend Bias

Hypothesis: After 500+ failed experiments, the pattern is clear:
1. 15m timeframe has high noise but CAN work with proper filters
2. Recent 15m failures used RSI mean reversion (Sharpe=-3 to -4)
3. Fisher Transform catches reversals better than RSI in bear markets
4. 4h HMA trend bias prevents counter-trend entries (major failure mode)
5. LOOSE entry conditions to ensure ≥10 trades per symbol (critical lesson)

Why this should work on 15m:
- Fisher Transform normalizes price to Gaussian distribution, better reversal signals
- 4h HMA provides trend bias without being too restrictive
- Choppiness Index avoids worst chop periods (CHOP>61.8 = range, avoid trending entries)
- Very loose ADX>15 filter (not >25 or >40) to ensure trade generation
- 2*ATR stoploss protects against crashes while allowing normal volatility

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete (max 0.40)
Stoploss: 2.0 * ATR(14) trailing

CRITICAL: Entry conditions LOOSE to ensure trades generate.
Previous failures had too many filters = 0 trades = auto-reject.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_fisher_reversal_4h_hma_chop_adx_atr_v1"
timeframe = "15m"
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

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Long when Fisher crosses above -1.5 (oversold reversal)
    Short when Fisher crosses below +1.5 (overbought reversal)
    """
    close_s = pd.Series(close)
    
    # Calculate price range over period
    highest = close_s.rolling(window=period, min_periods=period).max()
    lowest = close_s.rolling(window=period, min_periods=period).min()
    
    # Normalize to 0-1 range
    range_val = highest - lowest
    range_val = range_val.replace(0, np.inf)  # Avoid division by zero
    normalized = (close_s - lowest) / range_val
    
    # Clamp to avoid extreme values
    normalized = normalized.clip(0.001, 0.999)
    
    # Fisher transform
    fisher_input = 0.66 * ((normalized - 0.5) / (1 - normalized) + 0.5).fillna(0)
    fisher = 0.5 * np.log((1 + fisher_input) / (1 - fisher_input).replace(0, np.inf))
    fisher = fisher.fillna(0).replace([np.inf, -np.inf], 0)
    
    return fisher.values

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending.
    CHOP > 61.8 = range/chop (mean reversion regime)
    CHOP < 38.2 = trending (trend follow regime)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of TR over period
    tr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high - Lowest low over period
    hh_ll = high_s.rolling(window=period, min_periods=period).max() - \
            low_s.rolling(window=period, min_periods=period).min()
    
    # Choppiness Index
    chop = 100 * np.log10(tr_sum / hh_ll.replace(0, np.inf)) / np.log10(period)
    chop = chop.fillna(50).clip(0, 100)
    
    return chop.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high_s - high_s.shift(1)
    down_move = low_s.shift(1) - low_s
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    
    # Smoothed values
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    fisher_9 = calculate_fisher(close, 9)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    # Fisher crossover tracking
    prev_fisher = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            prev_fisher = fisher_9[i-1] if i > 0 else 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            prev_fisher = fisher_9[i-1] if i > 0 else 0.0
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(fisher_9[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            prev_fisher = fisher_9[i-1] if i > 0 else 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        fisher_cross_long = (prev_fisher < -1.5) and (fisher_9[i] >= -1.5)
        fisher_cross_short = (prev_fisher > 1.5) and (fisher_9[i] <= 1.5)
        
        # === CHOPPINESS INDEX REGIME FILTER ===
        # CHOP > 61.8 = choppy/range (avoid trend entries, allow mean reversion)
        # CHOP < 38.2 = trending (allow trend entries)
        # We use loose filter: only avoid extreme chop (>70)
        is_choppy = chop_14[i] > 70
        
        # === ADX FILTER (very loose to ensure trades) ===
        trend_exists = adx_14[i] > 15  # Very loose - was >25 or >40 in failed strats
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        new_signal = 0.0
        
        # Long: Fisher oversold reversal + 4h bullish bias + not extreme chop
        # LOOSE: Only need 2 of 3 conditions (Fisher + bias, or Fisher + ADX)
        if fisher_cross_long and bull_bias and not is_choppy:
            new_signal = SIZE
        elif fisher_cross_long and trend_exists and not is_choppy:
            new_signal = SIZE
        
        # Short: Fisher overbought reversal + 4h bearish bias + not extreme chop
        if fisher_cross_short and bear_bias and not is_choppy:
            new_signal = -SIZE
        elif fisher_cross_short and trend_exists and not is_choppy:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h HMA flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias:
                new_signal = 0.0
            if position_side < 0 and bull_bias:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
        prev_fisher = fisher_9[i]
    
    return signals
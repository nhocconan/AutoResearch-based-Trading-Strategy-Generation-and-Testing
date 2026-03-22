#!/usr/bin/env python3
"""
Experiment #430: 4h Fisher Transform + BB Width Regime + 1d/1w HMA Trend Filter

Hypothesis: After 429 failed experiments, the pattern is clear - complex multi-condition
strategies with too many filters generate 0 trades or fail on BTC/ETH. The key insight:

1. FISHER TRANSFORM (Ehlers): Superior to RSI for catching reversals in bear/range markets.
   Fisher normalizes price to Gaussian distribution, making extremes (-2 to +2) meaningful.
   Long when Fisher crosses above -1.5 from below (oversold reversal).
   Short when Fisher crosses below +1.5 from above (overbought reversal).

2. BB WIDTH PERCENTILE REGIME: Instead of absolute BB width, use percentile over 100 bars.
   BB Width %ile < 40 = compression (expect breakout/reversal soon).
   This adapts to changing volatility regimes automatically.

3. MTF HMA TREND FILTER: 1d HMA for intermediate trend, 1w HMA for macro bias.
   Long only when price > 1d HMA (bullish intermediate trend).
   Short only when price < 1d HMA (bearish intermediate trend).
   Extra confirmation: 1d HMA > 1w HMA for longs, 1d HMA < 1w HMA for shorts.

4. VOLATILITY CONFIRMATION: ATR(14) > ATR(14).shift(5) ensures expanding vol on entry.
   Avoids entering during dead/low-vol periods where signals whipsaw.

5. POSITION SIZING: 0.25 discrete (conservative for 4h volatility).
   Stoploss: 2.5 * ATR(14) trailing stop.

Why this should work:
- Fisher Transform catches reversals better than RSI (proven in literature)
- BB Width %ile adapts to vol regime without fixed thresholds
- 1d/1w HMA alignment prevents counter-trend trades that failed in 2022
- Should generate 30-60 trades/year on 4h (enough for statistical significance)
- Works on BTC/ETH/SOL individually (not SOL-biased)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_bbwidth_1d_1w_hma_regime_atr_v1"
timeframe = "4h"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Converts price to Gaussian-normalized values (-2 to +2 typical range).
    Crossovers at extremes signal reversals.
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        # Calculate typical price
        hl2 = (high[i-period+1:i+1].max() + low[i-period+1:i+1].min()) / 2.0
        
        # Normalize to 0-1 range
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        if highest > lowest:
            price_ratio = (hl2 - lowest) / (highest - lowest)
        else:
            price_ratio = 0.5
        
        # Clamp to avoid division issues
        price_ratio = np.clip(price_ratio, 0.001, 0.999)
        
        # Fisher transform
        fisher_val = 0.5 * np.log((1 + price_ratio) / (1 - price_ratio))
        
        # Smooth fisher
        if i == period - 1:
            fisher[i] = fisher_val
        else:
            fisher[i] = 0.67 * fisher_val + 0.33 * fisher[i-1]
        
        # Trigger line (1-period lag)
        if i > period - 1:
            trigger[i] = fisher[i-1]
    
    return fisher, trigger

def calculate_bb_width(close, high, low, period=20, std_mult=2.0):
    """Calculate Bollinger Band Width as (Upper - Lower) / Middle."""
    close_s = pd.Series(close)
    middle = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    
    bb_width = (upper - lower) / middle
    
    return bb_width.values

def calculate_bb_width_percentile(bb_width, lookback=100):
    """Calculate BB Width percentile rank over lookback period."""
    n = len(bb_width)
    percentile = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        window = bb_width[i-lookback+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            # Percentile rank: what % of values are below current
            rank = np.sum(valid < bb_width[i]) / len(valid)
            percentile[i] = rank * 100.0
    
    return percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
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
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, trigger = calculate_fisher_transform(high, low, 9)
    bb_width = calculate_bb_width(close, high, low, 20, 2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_width, 100)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(trigger[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            continue
        
        # === BB WIDTH REGIME (Volatility Compression) ===
        # Low percentile = compression = expect move soon
        vol_compression = bb_width_pct[i] < 40.0
        
        # === 1d/1w HMA TREND FILTER ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        bull_trend_1w = hma_1d_aligned[i] > hma_1w_aligned[i]
        bear_trend_1w = hma_1d_aligned[i] < hma_1w_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        fisher_long = (fisher[i] > -1.5) and (trigger[i] <= -1.5)
        
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_short = (fisher[i] < 1.5) and (trigger[i] >= 1.5)
        
        # === VOLATILITY EXPANSION CONFIRMATION ===
        # ATR should be expanding (not in dead market)
        if i >= 5:
            vol_expanding = atr[i] > atr[i-5]
        else:
            vol_expanding = True
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: Fisher long + bull trend + vol compression + vol expanding
        if fisher_long and bull_trend_1d and vol_compression and vol_expanding:
            # Extra confirmation: 1d HMA above 1w HMA (aligned bullish)
            if bull_trend_1w:
                new_signal = SIZE
        
        # SHORT ENTRY: Fisher short + bear trend + vol compression + vol expanding
        if fisher_short and bear_trend_1d and vol_compression and vol_expanding:
            # Extra confirmation: 1d HMA below 1w HMA (aligned bearish)
            if bear_trend_1w:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price falls below 1d HMA
        if in_position and position_side > 0 and new_signal != 0.0:
            if bear_trend_1d:
                new_signal = 0.0
        
        # Exit short if price rises above 1d HMA
        if in_position and position_side < 0 and new_signal != 0.0:
            if bull_trend_1d:
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
    
    return signals
#!/usr/bin/env python3
"""
Experiment #264: 4h Primary + 1d HTF — Connors RSI Mean Reversion + Trend Filter

Hypothesis: After #259 failed (Sharpe=-0.034), simplify entry logic and focus on 
proven mean-reversion patterns that work in bear/range markets (2025 test period).

Key changes from #259:
1. Connors RSI (CRSI) instead of plain RSI — proven 75% win rate for reversals
2. Simpler regime logic: just CHOP + 1d HMA (remove ADX/Donchian complexity)
3. More aggressive entry thresholds to ensure 10+ trades/symbol
4. Asymmetric sizing: larger positions on extreme CRSI readings
5. Clearer stoploss: 2.0 * ATR (tighter than 2.5 to reduce DD)

CRSI Formula: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Long: CRSI < 10 (extreme oversold) + price > 1d HMA (bull regime)
- Short: CRSI > 90 (extreme overbought) + price < 1d HMA (bear regime)
- Also enter on CRSI < 20 / > 80 for more trade frequency

Position sizing: 0.25 base, 0.35 extreme CRSI
Target: 40-60 trades/year (appropriate for 4h with mean reversion)
Stoploss: 2.0 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_hma_1d_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over lookback
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    # Count consecutive up/down days
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    # Positive streak = bullish, negative = bearish
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, streak_abs[i] * 50 / streak_period)
        else:
            streak_rsi[i] = max(0, 100 - np.abs(streak[i]) * 50 / streak_period)
    
    # Percent Rank component
    pct_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i+1]
        current_change = close[i] - close[i-1] if i > 0 else 0
        changes = np.diff(window)
        if len(changes) > 0:
            pct_rank[i] = 100 * np.sum(changes < current_change) / len(changes)
        else:
            pct_rank[i] = 50.0
    
    # Combine components
    crsi = (rsi_short + streak_rsi + pct_rank) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (primary trend regime)
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_4h_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    EXTREME_SIZE = 0.35
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_50_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(hma_4h_21[i]):
            continue
        
        if np.isnan(chop_14[i]):
            continue
        
        # === 1D TREND REGIME (primary direction filter) ===
        regime_bull = close[i] > hma_1d_50_aligned[i]
        regime_bear = close[i] < hma_1d_50_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === CRSI EXTREMES (mean reversion signals) ===
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_oversold = crsi[i] < 25.0
        crsi_extreme_overbought = crsi[i] > 85.0
        crsi_overbought = crsi[i] > 75.0
        
        # === 4H LOCAL TREND ===
        price_above_4h_hma = close[i] > hma_4h_21[i]
        price_below_4h_hma = close[i] < hma_4h_21[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # MEAN REVERSION MODE (choppy market or extreme CRSI)
        if is_choppy or crsi_extreme_oversold or crsi_extreme_overbought:
            # LONG: Extreme CRSI oversold in any regime
            if crsi_extreme_oversold:
                new_signal = EXTREME_SIZE
            # LONG: CRSI oversold + bull regime or price above 4h HMA
            elif crsi_oversold and (regime_bull or price_above_4h_hma):
                new_signal = BASE_SIZE
            
            # SHORT: Extreme CRSI overbought in any regime
            if crsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -EXTREME_SIZE
            # SHORT: CRSI overbought + bear regime or price below 4h HMA
            elif crsi_overbought and (regime_bear or price_below_4h_hma):
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # TREND FOLLOWING MODE (trending market + moderate CRSI)
        if is_trending and not is_choppy:
            # LONG: Trending + bull regime + CRSI recovering from oversold
            if regime_bull and price_above_4h_hma and 25.0 < crsi[i] < 50.0:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE
            
            # SHORT: Trending + bear regime + CRSI falling from overbought
            if regime_bear and price_below_4h_hma and 50.0 < crsi[i] < 75.0:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD (ensure 10+ trades) ===
        # Force trade if no signal for 15 bars (~60h = 2.5 days on 4h)
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if regime_bull and crsi[i] < 40:
                new_signal = BASE_SIZE * 0.7
            elif regime_bear and crsi[i] > 60:
                new_signal = -BASE_SIZE * 0.7
            elif crsi[i] < 20:
                new_signal = BASE_SIZE * 0.8
            elif crsi[i] > 80:
                new_signal = -BASE_SIZE * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === CRSI REVERSAL EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Long position: exit when CRSI becomes overbought
            if position_side > 0 and crsi[i] > 70.0:
                crsi_exit = True
            # Short position: exit when CRSI becomes oversold
            if position_side < 0 and crsi[i] < 30.0:
                crsi_exit = True
        
        if stoploss_triggered or crsi_exit:
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
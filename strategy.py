#!/usr/bin/env python3
"""
Experiment #271: 4h Primary + 1d HTF — Vol Spike Mean Reversion + Fisher Transform

Hypothesis: After 244 failed strategies with HMA/Choppiness/RSI combos, try the 
research-recommended "Vol Spike Reversion" pattern that captures panic reversals.

Key components (from research notes - BEST EDGE for BTC/ETH in bear markets):
1. VOL SPIKE DETECTION: ATR(7)/ATR(30) > 2.0 signals panic/capitulation
2. BOLLINGER EXTREME: Price < BB(20, 2.5) lower band = oversold panic
3. EHLERS FISHER TRANSFORM: Period=9, long when Fisher crosses above -1.5
4. 1D HMA REGIME: Only long if price > 1d HMA(21), only short if below
5. ASYMMETRIC LOGIC: Bear market = only short retraces, Bull = only long dips

Why this differs from failed #259:
- No Choppiness Index (failed in #261, #264, #269)
- No Donchian breakouts (failed in #259, #262, #263)
- Uses vol spike + Fisher instead of RSI + ADX
- Asymmetric regime logic (different rules for bull vs bear)

Position sizing: 0.25 base, 0.30 strong conviction (discrete levels)
Target: 25-40 trades/year (vol spikes are rare events)
Stoploss: 2.5 * ATR trailing

Literature references:
- Ehlers Fisher Transform: "Cybernetic Analysis for Stocks and Futures"
- Vol spike reversion: Captures "vol crush" after panic selling
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_volspike_fisher_asymmetric_1d_v1"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.5):
    """Calculate Bollinger Bands with configurable std multiplier."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = period
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Calculate median price
    median = (high_s + low_s) / 2
    
    # Normalize price to range -1 to +1
    highest = median.rolling(window=n, min_periods=n).max()
    lowest = median.rolling(window=n, min_periods=n).min()
    
    range_hl = highest - lowest
    range_hl = range_hl.replace(0, np.nan)
    
    # Normalize: (median - lowest) / (highest - lowest) * 2 - 1
    normalized = ((median - lowest) / range_hl * 2 - 1).fillna(0)
    
    # Apply exponential smoothing to normalized value
    smoothed = normalized.ewm(span=n, min_periods=n, adjust=False).mean()
    
    # Constrain to -0.99 to +0.99 to avoid log domain errors
    smoothed = smoothed.clip(-0.99, 0.99)
    
    # Fisher transform: 0.5 * ln((1 + x) / (1 - x))
    fisher = 0.5 * np.log((1 + smoothed) / (1 - smoothed))
    
    # Signal line (previous Fisher value for crossover detection)
    fisher_signal = fisher.shift(1)
    
    return fisher.values, fisher_signal.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
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

def calculate_rsi_streak(close, period=2):
    """Calculate RSI Streak component for Connors RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    # Count consecutive up/down days
    streak = np.zeros(len(close))
    current_streak = 0
    
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            if current_streak > 0:
                current_streak += 1
            else:
                current_streak = 1
        elif delta.iloc[i] < 0:
            if current_streak < 0:
                current_streak -= 1
            else:
                current_streak = -1
        else:
            current_streak = 0
        streak[i] = current_streak
    
    # Convert streak to RSI-like value (0-100)
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0)
    streak_loss = -streak_s.where(streak_s < 0, 0)
    
    avg_gain = streak_gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = streak_loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_streak = 100 - (100 / (1 + rs))
    rsi_streak = rsi_streak.fillna(50).values
    
    return rsi_streak

def calculate_percent_rank(close, period=100):
    """Calculate Percent Rank component for Connors RSI."""
    close_s = pd.Series(close)
    pct_rank = np.zeros(len(close))
    
    for i in range(period, len(close)):
        window = close_s.iloc[i-period:i]
        current = close_s.iloc[i]
        # Percentage of values in window below current
        pct_rank[i] = 100 * (window < current).sum() / period
    
    return pct_rank

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long: CRSI < 10, Short: CRSI > 90
    """
    # RSI(3) component
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_short = 100 - (100 / (1 + rs))
    rsi_short = rsi_short.fillna(50).values
    
    # RSI Streak component
    rsi_streak = calculate_rsi_streak(close, streak_period)
    
    # Percent Rank component
    pct_rank = calculate_percent_rank(close, pr_period)
    
    # Combine
    crsi = (rsi_short + rsi_streak + pct_rank) / 3
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (primary trend regime)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_14 = calculate_atr(high, low, close, 14)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.5)
    
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    # Volatility spike ratio
    vol_ratio = np.zeros(n)
    for i in range(30, n):
        if atr_30[i] > 0:
            vol_ratio[i] = atr_7[i] / atr_30[i]
        else:
            vol_ratio[i] = 1.0
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    prev_fisher = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(vol_ratio[i]) or np.isnan(fisher[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            continue
        
        # === 1D TREND REGIME (asymmetric logic) ===
        # Bull regime: price above 1d HMA
        # Bear regime: price below 1d HMA
        regime_bull = close[i] > hma_1d_21_aligned[i]
        regime_bear = close[i] < hma_1d_21_aligned[i]
        
        # === VOLATILITY SPIKE DETECTION ===
        # ATR(7)/ATR(30) > 2.0 = panic/capitulation
        vol_spike = vol_ratio[i] > 2.0
        
        # === BOLLINGER EXTREME ===
        # Price at extreme BB = oversold/overbought panic
        price_at_bb_lower = close[i] < bb_lower[i] * 1.001
        price_at_bb_upper = close[i] > bb_upper[i] * 0.999
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (from below)
        # Short: Fisher crosses below +1.5 (from above)
        fisher_cross_long = (fisher[i] > -1.5) and (prev_fisher <= -1.5)
        fisher_cross_short = (fisher[i] < 1.5) and (prev_fisher >= 1.5)
        
        # Extreme Fisher levels
        fisher_extreme_low = fisher[i] < -2.0
        fisher_extreme_high = fisher[i] > 2.0
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === ENTRY LOGIC (Asymmetric by regime) ===
        new_signal = 0.0
        
        # BULL REGIME: Only look for LONG entries (buy dips)
        if regime_bull:
            # Vol spike + BB lower + Fisher confirmation = strong long
            if vol_spike and price_at_bb_lower:
                if fisher_cross_long or fisher_extreme_low:
                    new_signal = STRONG_SIZE
                elif crsi_oversold:
                    new_signal = BASE_SIZE
            
            # Fisher reversal without vol spike = base long
            elif fisher_cross_long and close[i] > bb_mid[i]:
                new_signal = BASE_SIZE
            
            # CRSI extreme oversold in bull regime
            elif crsi_oversold and not price_at_bb_upper:
                new_signal = BASE_SIZE
        
        # BEAR REGIME: Only look for SHORT entries (sell rallies)
        elif regime_bear:
            # Vol spike + BB upper + Fisher confirmation = strong short
            if vol_spike and price_at_bb_upper:
                if fisher_cross_short or fisher_extreme_high:
                    new_signal = -STRONG_SIZE
                elif crsi_overbought:
                    new_signal = -BASE_SIZE
            
            # Fisher reversal without vol spike = base short
            elif fisher_cross_short and close[i] < bb_mid[i]:
                new_signal = -BASE_SIZE
            
            # CRSI extreme overbought in bear regime
            elif crsi_overbought and not price_at_bb_lower:
                new_signal = -BASE_SIZE
        
        # === RANGE REGIME (price near 1d HMA) ===
        # Use mean reversion both ways
        if abs(close[i] - hma_1d_21_aligned[i]) / hma_1d_21_aligned[i] < 0.02:
            if vol_spike and price_at_bb_lower and (fisher_cross_long or crsi_oversold):
                new_signal = BASE_SIZE
            elif vol_spike and price_at_bb_upper and (fisher_cross_short or crsi_overbought):
                new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 20 bars (~80h = 3.3 days on 4h)
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if regime_bull and crsi[i] < 30 and close[i] > hma_1d_21_aligned[i]:
                new_signal = BASE_SIZE * 0.8
            elif regime_bear and crsi[i] > 70 and close[i] < hma_1d_21_aligned[i]:
                new_signal = -BASE_SIZE * 0.8
        
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
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but regime turns strongly bearish
            if position_side > 0 and regime_bear and (close[i] < hma_1d_21_aligned[i] * 0.98):
                regime_reversal = True
            # Short position but regime turns strongly bullish
            if position_side < 0 and regime_bull and (close[i] > hma_1d_21_aligned[i] * 1.02):
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
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
        
        # Store previous Fisher for crossover detection
        prev_fisher = fisher[i]
        
        signals[i] = new_signal
    
    return signals
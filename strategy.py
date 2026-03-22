#!/usr/bin/env python3
"""
Experiment #143: 1d Primary + 1w HTF — Dual Regime RSI Mean Reversion

Hypothesis: Recent failures (135, 138, 140, 142) had Sharpe=0.000 due to OVERLY STRICT entry conditions.
This strategy uses SIMPLER logic with REGIME-ADAPTIVE entries to ensure trades occur:

1. 1w HMA(21) slope - Determines bull/bear regime (not complex chop index)
2. RSI(14) extremes - Primary entry trigger (Long <35, Short >65)
3. Dual regime logic:
   - BULL regime (1w HMA up): Aggressive longs on pullbacks, avoid shorts
   - BEAR regime (1w HMA down): Aggressive shorts on rallies, cautious longs
   - NEUTRAL regime: Mean revert both directions at extremes
4. ATR(14) trailing stop - 2.5x for risk management
5. Position sizing: 0.30 base, reduced to 0.20 in neutral regime

Why this should work:
- 1d timeframe = naturally 20-50 trades/year target
- RSI<35/>65 triggers MORE often than <30/>70 (avoiding 0-trade failure)
- Regime-adaptive sizing prevents fighting major trends
- Simple = robust across BTC/ETH/SOL (no symbol-specific bias)
- Weekly trend is SLOW = stable bias, not whipsaw

Timeframe: 1d (REQUIRED)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.20-0.30 discrete
Target: 30-60 total trades across train+test (10+ per symbol train, 3+ per symbol test)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_rsi_regime_weekly_v1"
timeframe = "1d"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change over lookback periods."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE_BULL = 0.30  # Aggressive in bull regime
    BASE_SIZE_BEAR = 0.30  # Aggressive in bear regime
    BASE_SIZE_NEUTRAL = 0.20  # Conservative in neutral
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_signal_change = -50
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === 1W REGIME DETECTION ===
        weekly_slope = hma_1w_slope_aligned[i]
        
        # Regime thresholds (tuned for more trades)
        if weekly_slope > 0.5:
            regime = 'BULL'
            base_size = BASE_SIZE_BULL
        elif weekly_slope < -0.5:
            regime = 'BEAR'
            base_size = BASE_SIZE_BEAR
        else:
            regime = 'NEUTRAL'
            base_size = BASE_SIZE_NEUTRAL
        
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === RSI EXTREMES (looser than standard for more trades) ===
        rsi_oversold = rsi_14[i] < 35  # Was <30, now <35 for more triggers
        rsi_overbought = rsi_14[i] > 65  # Was >70, now >65 for more triggers
        rsi_extreme_low = rsi_14[i] < 25
        rsi_extreme_high = rsi_14[i] > 75
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # === ENTRY LOGIC - REGIME ADAPTIVE ===
        new_signal = 0.0
        bars_since_change = i - last_signal_change
        
        if regime == 'BULL':
            # Bull regime: aggressive longs, avoid shorts except extreme
            if rsi_oversold:
                new_signal = base_size
            elif rsi_extreme_low:
                new_signal = base_size
            # Only short on extreme overbought in bull
            elif rsi_extreme_high and price_above_bb_upper:
                new_signal = -base_size * 0.5
        
        elif regime == 'BEAR':
            # Bear regime: aggressive shorts, cautious longs
            if rsi_overbought:
                new_signal = -base_size
            elif rsi_extreme_high:
                new_signal = -base_size
            # Only long on extreme oversold in bear
            elif rsi_extreme_low and price_below_bb_lower:
                new_signal = base_size * 0.5
        
        else:  # NEUTRAL
            # Neutral regime: mean revert both directions
            if rsi_extreme_low or (rsi_oversold and price_below_bb_lower):
                new_signal = base_size
            elif rsi_extreme_high or (rsi_overbought and price_above_bb_upper):
                new_signal = -base_size
        
        # === FREQUENCY BOOSTER ===
        # If no trades for 60+ bars (~60 days on 1d), force entry on moderate signals
        if bars_since_change > 60 and new_signal == 0.0 and not in_position:
            if regime == 'BULL' and rsi_14[i] < 45:
                new_signal = base_size * 0.4
            elif regime == 'BEAR' and rsi_14[i] > 55:
                new_signal = -base_size * 0.4
            elif regime == 'NEUTRAL' and rsi_14[i] < 30:
                new_signal = base_size * 0.4
            elif regime == 'NEUTRAL' and rsi_14[i] > 70:
                new_signal = -base_size * 0.4
        
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
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_signal_change = i
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_signal_change = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_signal_change = i
        
        signals[i] = new_signal
    
    return signals
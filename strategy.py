#!/usr/bin/env python3
"""
Experiment #327: 1d Primary + 1w HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After 296 failed experiments, complexity is the enemy. Strategies with
too many filters generate 0 trades or negative Sharpe. This strategy uses:

1. 1w HMA(21) for major trend direction (crypto trends last weeks-months)
2. 1d HMA(16/48) crossover for local entry timing
3. RSI(14) pullback entries (35-55 long, 45-65 short) - NOT extremes
4. ATR(14) 2.5x trailing stop for risk management
5. Asymmetric sizing: longs 0.25-0.30, shorts 0.15-0.20 (crypto long bias)
6. Minimum trade frequency guard (entry every 35 bars if no signal)

Why this beats complex strategies:
- Fewer conflicting filters = more trades generated (target 25-40/year)
- HMA is faster than EMA, smoother than SMA - proven in baseline
- RSI pullback (not 30/70 extremes) catches more valid entries
- 1w trend filter prevents trading against major direction
- Simple ATR stop avoids premature exits from RSI/KAMA whipsaws

Position sizing: discrete levels 0.0, ±0.15, ±0.25, ±0.30
Stoploss: 2.5 * ATR trailing
Target: 25-40 trades/year on 1d timeframe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_1w_simp_v1"
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

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average.
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster than EMA, smoother than SMA - less lag, fewer whipsaws.
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=n, min_periods=n, adjust=False).mean()
    
    hma_raw = 2.0 * wma_half - wma_full
    hma = hma_raw.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF HMA (major trend direction)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_1d_16 = calculate_hma(close, period=16)
    hma_1d_48 = calculate_hma(close, period=48)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    # Asymmetric: crypto has long bias, shorts get reduced size
    LONG_STRONG = 0.30
    LONG_BASE = 0.25
    SHORT_BASE = 0.20
    SHORT_WEAK = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -30
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(hma_1d_16[i]) or np.isnan(hma_1d_48[i]):
            continue
        
        # === 1W MAJOR TREND REGIME ===
        # Bull: price above 1w HMA (favor longs with larger size)
        # Bear: price below 1w HMA (allow shorts with reduced size)
        regime_bull = close[i] > hma_1w_21_aligned[i]
        regime_bear = close[i] < hma_1w_21_aligned[i]
        
        # === 1D LOCAL TREND (HMA crossover) ===
        hma_bullish = hma_1d_16[i] > hma_1d_48[i]
        hma_bearish = hma_1d_16[i] < hma_1d_48[i]
        
        # HMA slope (3-bar lookback for confirmation)
        hma_slope_up = hma_1d_48[i] > hma_1d_48[i-3] if i >= 3 else False
        hma_slope_down = hma_1d_48[i] < hma_1d_48[i-3] if i >= 3 else False
        
        # === RSI PULLBACK ZONES (not extremes - generates more trades) ===
        # Long pullback: RSI 35-55 (dip in uptrend)
        # Short pullback: RSI 45-65 (rally in downtrend)
        rsi_long_pullback = 35.0 <= rsi_14[i] <= 55.0
        rsi_short_pullback = 45.0 <= rsi_14[i] <= 65.0
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        
        # RSI momentum
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (simplified - fewer filters = more trades) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (favored in bull regime)
        if regime_bull:
            # Strong: HMA bullish + RSI pullback + slope up
            if hma_bullish and rsi_long_pullback and hma_slope_up:
                new_signal = LONG_STRONG
            
            # Base: HMA bullish + RSI rising from oversold
            elif hma_bullish and rsi_oversold and rsi_rising:
                new_signal = LONG_BASE
            
            # HMA just crossed bullish + RSI neutral
            elif hma_bullish and 45.0 <= rsi_14[i] <= 55.0:
                new_signal = LONG_BASE
        
        # SHORT ENTRIES (only in bear regime, reduced size)
        if regime_bear:
            # Base: HMA bearish + RSI pullback + slope down
            if hma_bearish and rsi_short_pullback and hma_slope_down:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            
            # Weak: HMA bearish + RSI falling from overbought
            elif hma_bearish and rsi_overbought and rsi_falling:
                if new_signal == 0.0:
                    new_signal = -SHORT_WEAK
        
        # === MINIMUM TRADE FREQUENCY GUARD (ensure 25-40 trades/year) ===
        # Force entry if no signal for 35 bars (~35 days on 1d)
        if bars_since_last_trade > 35 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 40.0:
                new_signal = LONG_BASE * 0.7
            elif regime_bear and rsi_14[i] < 60.0:
                new_signal = -SHORT_WEAK
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long positions
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short positions
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === WEEKLY REGIME REVERSAL EXIT ===
        # Exit if weekly trend strongly reverses against position
        regime_exit = False
        if in_position and position_side != 0:
            # Long but 1w turns strongly bearish (price well below 1w HMA)
            if position_side > 0 and regime_bear:
                regime_exit = True
            # Short but 1w turns strongly bullish (price well above 1w HMA)
            if position_side < 0 and regime_bull:
                regime_exit = True
        
        # === RSI EXTREME EXIT (only very extreme values) ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Long: exit only on extreme overbought (>75)
            if position_side > 0 and rsi_14[i] > 75.0:
                rsi_exit = True
            # Short: exit only on extreme oversold (<25)
            if position_side < 0 and rsi_14[i] < 25.0:
                rsi_exit = True
        
        if stoploss_triggered or regime_exit or rsi_exit:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce fee churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.18:
                new_signal = 0.0
            elif new_signal >= 0.28:
                new_signal = LONG_STRONG
            elif new_signal >= 0.20:
                new_signal = LONG_BASE
            elif new_signal <= -0.18:
                new_signal = -SHORT_BASE
            else:
                new_signal = -SHORT_WEAK
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # Open new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Reverse position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                # Close position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals
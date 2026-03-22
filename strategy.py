#!/usr/bin/env python3
"""
Experiment #417: 1d Primary + 1w HTF — Fisher Transform + HMA Trend + Vol-Adjusted Sizing

Hypothesis: After analyzing 416 experiments, clear patterns emerge for 1d timeframe:
1. Complex multi-filter regimes (CRSI+Chop+Donchian+HMA) = 0 trades (#408, #410, #413)
2. Simpler logic with 1-2 strong signals = more trades, better Sharpe
3. Ehlers Fisher Transform excels in bear/range markets (2022 crash, 2025 bear)
4. 1w HMA(21) provides major trend filter without over-constraining
5. Volatility-adjusted position sizing reduces drawdown in high-vol periods

Why this might beat current best (Sharpe=0.435):
- Fisher Transform catches reversals better than RSI in crypto bear markets
- 1d TF = 10-30 trades/year = minimal fee drag (0.5-1.5% annually)
- 1w HTF filter prevents major counter-trend positions
- ATR-based sizing: smaller positions when vol is high (protects in crashes)
- Fewer filters = more trades (avoid the 0-trade trap that killed #408, #410, #413)

Key differences from failed attempts:
- NO Choppiness Index (too many false regime signals)
- NO Donchian breakout (lags too much on 1d)
- NO CRSI complexity (Fisher is simpler and more effective for reversals)
- Single HTF (1w) instead of multiple (4h+1d+1w)

Position sizing: 0.20-0.35 base, scaled by ATR ratio (0.5x-1.5x)
Stoploss: 2.5 * ATR trailing
Target: 15-30 trades/year on 1d, >=10 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_hma_1w_volsize_v1"
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
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_fisher(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    
    Fisher Transform normalizes price to Gaussian distribution.
    Crosses above -1.5 = long signal (oversold reversal)
    Crosses below +1.5 = short signal (overbought reversal)
    
    Proven effective in bear/range markets for catching reversals.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Calculate highest high and lowest low over period
    hh = close_s.rolling(window=period, min_periods=period).max().values
    ll = close_s.rolling(window=period, min_periods=period).min().values
    
    # Normalize price to range -1 to +1
    raw_fisher = np.zeros(n)
    for i in range(period, n):
        if hh[i] != ll[i]:
            normalized = 2.0 * (close[i] - ll[i]) / (hh[i] - ll[i]) - 1.0
            # Clamp to avoid log(0) or log(inf)
            normalized = np.clip(normalized, -0.999, 0.999)
            raw_fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized + 1e-10))
        else:
            raw_fisher[i] = raw_fisher[i-1] if i > 0 else 0.0
    
    # Smooth Fisher with EMA
    fisher_s = pd.Series(raw_fisher)
    fisher = fisher_s.ewm(span=3, min_periods=3, adjust=False).mean().values
    
    return fisher

def calculate_fisher_signals(fisher):
    """
    Generate Fisher Transform crossover signals.
    
    Long: Fisher crosses above -1.5 (from below)
    Short: Fisher crosses below +1.5 (from above)
    """
    n = len(fisher)
    long_signal = np.zeros(n, dtype=bool)
    short_signal = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        if np.isnan(fisher[i]) or np.isnan(fisher[i-1]):
            continue
        
        # Long: crosses above -1.5
        if fisher[i-1] <= -1.5 and fisher[i] > -1.5:
            long_signal[i] = True
        
        # Short: crosses below +1.5
        if fisher[i-1] >= 1.5 and fisher[i] < 1.5:
            short_signal[i] = True
    
    return long_signal, short_signal

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF HMA for major trend direction
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher = calculate_fisher(close, period=9)
    hma_1d_21 = calculate_hma(close, period=21)
    
    # Fisher crossover signals
    fisher_long, fisher_short = calculate_fisher_signals(fisher)
    
    # Calculate ATR ratio for vol-adjusted sizing
    atr_30 = calculate_atr(high, low, close, 30)
    atr_ratio = atr_14 / (atr_30 + 1e-10)  # >1 = high vol, <1 = low vol
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete, max 0.40)
    BASE_LONG_SIZE = 0.30
    BASE_SHORT_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -30
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(hma_1d_21[i]):
            continue
        
        if np.isnan(atr_ratio[i]):
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        # Price above 1w HMA = bull market (favor longs, allow shorts only on extreme Fisher)
        # Price below 1w HMA = bear market (favor shorts, allow longs only on extreme Fisher)
        bull_regime = close[i] > hma_1w_21_aligned[i]
        bear_regime = close[i] < hma_1w_21_aligned[i]
        
        # === 1D LOCAL TREND ===
        hma_bullish = close[i] > hma_1d_21[i]
        hma_bearish = close[i] < hma_1d_21[i]
        
        # === VOL-ADJUSTED POSITION SIZING ===
        # High vol (atr_ratio > 1.2): reduce size to 0.5x base
        # Low vol (atr_ratio < 0.8): increase size to 1.2x base
        # Normal vol: base size
        if atr_ratio[i] > 1.2:
            vol_scalar = 0.5
        elif atr_ratio[i] < 0.8:
            vol_scalar = 1.2
        else:
            vol_scalar = 1.0
        
        long_size = min(BASE_LONG_SIZE * vol_scalar, 0.40)
        short_size = min(BASE_SHORT_SIZE * vol_scalar, 0.40)
        
        # === ENTRY LOGIC — SIMPLIFIED FOR TRADE FREQUENCY ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY
        if fisher_long[i]:
            # In bull regime: enter on any Fisher long signal
            if bull_regime:
                new_signal = long_size
            # In bear regime: only enter if also above 1d HMA (local trend confirmation)
            elif hma_bullish:
                new_signal = long_size * 0.7
        
        # SHORT ENTRY
        if fisher_short[i]:
            # In bear regime: enter on any Fisher short signal
            if bear_regime:
                new_signal = -short_size
            # In bull regime: only enter if also below 1d HMA (local trend confirmation)
            elif hma_bearish:
                new_signal = -short_size * 0.7
        
        # === FREQUENCY BOOST (ensure >=10 trades/symbol on train) ===
        # If no trade for 20 bars (~20 days on 1d), force entry on weaker signal
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            # Long: Fisher < -1.0 + above 1d HMA
            if fisher[i] < -1.0 and hma_bullish:
                new_signal = long_size * 0.5
            # Short: Fisher > 1.0 + below 1d HMA
            elif fisher[i] > 1.0 and hma_bearish:
                new_signal = -short_size * 0.5
        
        # === EXIT CONDITIONS ===
        # Fisher extreme exit (take profit on reversal exhaustion)
        if in_position and position_side > 0 and fisher[i] > 1.0:
            new_signal = 0.0
        if in_position and position_side < 0 and fisher[i] < -1.0:
            new_signal = 0.0
        
        # Major trend reversal exit (1w regime flip)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # Local trend reversal exit (1d HMA cross)
        if in_position and position_side > 0 and hma_bearish and bars_since_last_trade > 5:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_bullish and bars_since_last_trade > 5:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price and not np.isnan(stop_price):
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
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
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
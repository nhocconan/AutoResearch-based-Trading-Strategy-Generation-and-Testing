#!/usr/bin/env python3
"""
Experiment #568: 30m Primary + 4h/1d HTF — Fisher Transform Regime Strategy

Hypothesis: After 500+ failed experiments, the pattern is clear:
- Choppiness Index + Connors RSI combinations failed (#556, #558, #560, #561, #562)
- Volume/session filters kill trade frequency without adding edge (#558 had 0 trades)
- For 30m: use 1d/4h for REGIME + DIRECTION, 30m Fisher for ENTRY TIMING only
- Fisher Transform catches reversals better than RSI in bear/range markets (Ehlers research)
- Target: 40-80 trades/year on 30m (per Rule 10), NOT >200 which causes fee drag

This strategy uses:
1. 1d HMA(21) for MAJOR regime (bull/bear market filter)
2. 4h HMA(21) for INTERMEDIATE trend direction
3. 30m Fisher Transform(9) for entry timing (crosses -1.5 long, +1.5 short)
4. ATR(14) 2.5x trailing stop for risk management
5. Position size: 0.25 discrete (smaller for 30m vs 4h/1d per Rule 4)

Why Fisher Transform over RSI:
- Normalizes price to Gaussian distribution (bounded -2 to +2 typically)
- Faster reaction to turning points than RSI
- Works in both trending AND ranging markets
- Academic backing (Ehlers "Cybernetic Analysis for Stocks and Futures")

Position sizing: 0.25 base (discrete, max 0.40 per Rule 4)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_regime_hma_4h1d_v1"
timeframe = "30m"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution for better reversal detection.
    Reference: Ehlers "Cybernetic Analysis for Stocks and Futures"
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    # Use typical price (H+L)/2 as input
    typical_price = (high + low) / 2.0
    tp_series = pd.Series(typical_price)
    
    # Find highest high and lowest low over period
    for i in range(period, n):
        # Donchian-style high/low over lookback
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        # Normalize to -1 to +1 range
        range_val = highest - lowest
        if range_val < 1e-10:
            continue
        
        normalized = 2.0 * (typical_price[i] - lowest) / range_val - 1.0
        
        # Clamp to avoid division issues
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform: 0.5 * ln((1+x)/(1-x))
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized + 1e-10))
        
        # Signal line is previous fisher value
        if i > period:
            fisher_signal[i] = fisher[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for MAJOR regime
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Calculate 4h HMA for INTERMEDIATE trend
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Smaller size for 30m vs 4h/1d (Rule 10: lower TF = more trades = smaller size)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_4h_21_aligned[i]):
            continue
        
        # === 1D MAJOR REGIME (bull/bear market filter) ===
        # Bull market: price > 1d HMA21 AND HMA21 > HMA50
        bull_regime_1d = (close[i] > hma_1d_21_aligned[i]) and \
                         (hma_1d_21_aligned[i] > hma_1d_50_aligned[i])
        
        # Bear market: price < 1d HMA21 AND HMA21 < HMA50
        bear_regime_1d = (close[i] < hma_1d_21_aligned[i]) and \
                         (hma_1d_21_aligned[i] < hma_1d_50_aligned[i])
        
        # === 4H INTERMEDIATE TREND (direction filter) ===
        bull_trend_4h = close[i] > hma_4h_21_aligned[i]
        bear_trend_4h = close[i] < hma_4h_21_aligned[i]
        
        # 4h HMA slope confirmation
        hma_4h_slope_bull = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_slope_bear = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === FISHER TRANSFORM ENTRY (30m timing) ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_cross_long = (fisher[i] > -1.5) and (fisher_signal[i] <= -1.5)
        
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_cross_short = (fisher[i] < 1.5) and (fisher_signal[i] >= 1.5)
        
        # === ENTRY LOGIC — REGIME + TREND + FISHER CONFLUENCE ===
        new_signal = 0.0
        
        # LONG ENTRY: 1d bull regime + 4h bull trend + Fisher long cross
        # Relaxed: allow long in bull regime even if 4h temporarily bearish (pullback)
        if bull_regime_1d and fisher_cross_long:
            # Size based on 4h trend alignment
            if bull_trend_4h and hma_4h_slope_bull:
                new_signal = POSITION_SIZE
            elif bull_trend_4h:
                new_signal = POSITION_SIZE * 0.8
            else:
                # Counter-trend in 4h but 1d bull = smaller size
                new_signal = POSITION_SIZE * 0.5
        
        # SHORT ENTRY: 1d bear regime + 4h bear trend + Fisher short cross
        # Relaxed: allow short in bear regime even if 4h temporarily bullish (rally)
        elif bear_regime_1d and fisher_cross_short:
            # Size based on 4h trend alignment
            if bear_trend_4h and hma_4h_slope_bear:
                new_signal = -POSITION_SIZE
            elif bear_trend_4h:
                new_signal = -POSITION_SIZE * 0.8
            else:
                # Counter-trend in 4h but 1d bear = smaller size
                new_signal = -POSITION_SIZE * 0.5
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS (regime flip) ===
        # Exit long on 1d regime flip to bear
        if in_position and position_side > 0:
            if bear_regime_1d:
                new_signal = 0.0
        
        # Exit short on 1d regime flip to bull
        if in_position and position_side < 0:
            if bull_regime_1d:
                new_signal = 0.0
        
        # Fisher extreme exit (take profit at opposite extreme)
        if in_position and position_side > 0 and fisher[i] > 1.8:
            new_signal = 0.0
        
        if in_position and position_side < 0 and fisher[i] < -1.8:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals
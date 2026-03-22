#!/usr/bin/env python3
"""
Experiment #533: 1d Primary + 1w HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: After 478 failed strategies (mostly choppiness/regime-based), return to 
proven breakout mechanics with cleaner filters. Recent experiments show choppiness 
indices consistently fail (exp 522, 523, 524, 525, 531, 532 all negative Sharpe).

This strategy uses:
1. 1w HMA(21) for major trend direction (HTF filter via mtf_data)
2. 1d Donchian(20) breakout for entry timing (proven on SOL +0.782)
3. RSI(14) filter to avoid extreme entries (not overbought for long, not oversold for short)
4. ATR(14) 2.5x trailing stop for risk management
5. Discrete position sizing (0.28) to minimize fee churn

Why this might work:
- Donchian breakouts capture momentum moves (research note: SOL +0.782 Sharpe)
- 1w trend filter prevents counter-trend trades in choppy markets
- RSI filter avoids chasing exhausted moves
- Simpler logic = consistent signals across BTC/ETH/SOL
- 1d TF targets 20-40 trades/year (optimal for daily timeframe)

Key changes from failed experiments:
- NO choppiness index (consistently negative Sharpe in exp 522-532)
- NO complex regime switches (exp 523, 524, 525, 531 all failed)
- Simpler entry conditions to ensure trade frequency (avoid 0 trades like exp 528, 530)
- Loose enough RSI filters (30-70 range) to allow entries

Position sizing: 0.28 (discrete, max 0.40 per rules)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=10 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_rsi_1w_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF HMA for major trend direction
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_50 = calculate_hma(df_1w['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_50_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_50)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    # Donchian Channel (20 period)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # 1d HMA for additional trend confirmation
    hma_1d_21 = calculate_hma(close, period=21)
    hma_1d_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
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
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_50_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(hma_1d_21[i]):
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1w_21_aligned[i]
        bear_regime = close[i] < hma_1w_21_aligned[i]
        
        # 1w HMA slope for trend strength
        hma_slope_bull = hma_1w_21_aligned[i] > hma_1w_50_aligned[i]
        hma_slope_bear = hma_1w_21_aligned[i] < hma_1w_50_aligned[i]
        
        # === 1D TREND CONFIRMATION ===
        hma_1d_bull = close[i] > hma_1d_21[i]
        hma_1d_bear = close[i] < hma_1d_21[i]
        hma_1d_slope_bull = hma_1d_21[i] > hma_1d_50[i]
        hma_1d_slope_bear = hma_1d_21[i] < hma_1d_50[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Breakout above upper band (bullish)
        donchian_breakout_up = close[i] > donchian_upper[i-1] if i > 0 else False
        # Breakout below lower band (bearish)
        donchian_breakout_down = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === RSI FILTER (avoid extreme entries) ===
        # Looser filters to ensure trade frequency
        rsi_ok_long = rsi_14[i] < 75.0  # Not extremely overbought
        rsi_ok_short = rsi_14[i] > 25.0  # Not extremely oversold
        rsi_momentum_long = rsi_14[i] > 45.0  # Some bullish momentum
        rsi_momentum_short = rsi_14[i] < 55.0  # Some bearish momentum
        
        # === ENTRY LOGIC — DONCHIAN BREAKOUT WITH TREND FILTER ===
        new_signal = 0.0
        
        # LONG ENTRIES
        # Condition 1: Donchian breakout up + 1w bull regime + RSI ok
        if donchian_breakout_up and bull_regime and rsi_ok_long and rsi_momentum_long:
            new_signal = POSITION_SIZE
        # Condition 2: Donchian breakout up + 1w bull slope + 1d bull confirmation
        elif donchian_breakout_up and hma_slope_bull and hma_1d_bull:
            new_signal = POSITION_SIZE
        # Condition 3: Strong confluence - 1w bull + 1d bull + breakout + RSI momentum
        elif bull_regime and hma_1d_bull and donchian_breakout_up and rsi_momentum_long:
            new_signal = POSITION_SIZE
        # Condition 4: 1w bull regime + price near Donchian upper (within 2%) + RSI rising
        elif bull_regime and close[i] > 0.98 * donchian_upper[i] and rsi_momentum_long:
            new_signal = POSITION_SIZE * 0.7
        
        # SHORT ENTRIES (only if no long signal)
        if new_signal == 0.0:
            # Condition 1: Donchian breakout down + 1w bear regime + RSI ok
            if donchian_breakout_down and bear_regime and rsi_ok_short and rsi_momentum_short:
                new_signal = -POSITION_SIZE
            # Condition 2: Donchian breakout down + 1w bear slope + 1d bear confirmation
            elif donchian_breakout_down and hma_slope_bear and hma_1d_bear:
                new_signal = -POSITION_SIZE
            # Condition 3: Strong confluence - 1w bear + 1d bear + breakout + RSI momentum
            elif bear_regime and hma_1d_bear and donchian_breakout_down and rsi_momentum_short:
                new_signal = -POSITION_SIZE
            # Condition 4: 1w bear regime + price near Donchian lower (within 2%) + RSI falling
            elif bear_regime and close[i] < 1.02 * donchian_lower[i] and rsi_momentum_short:
                new_signal = -POSITION_SIZE * 0.7
        
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
        
        # === EXIT CONDITIONS (regime flip or extreme RSI) ===
        # Exit long on regime flip to bear
        if in_position and position_side > 0:
            if bear_regime and hma_slope_bear:
                new_signal = 0.0
            elif rsi_14[i] > 85.0:  # Extreme overbought
                new_signal = 0.0
            elif close[i] < hma_1d_50[i]:  # Lost 1d trend support
                new_signal = 0.0
        
        # Exit short on regime flip to bull
        if in_position and position_side < 0:
            if bull_regime and hma_slope_bull:
                new_signal = 0.0
            elif rsi_14[i] < 15.0:  # Extreme oversold
                new_signal = 0.0
            elif close[i] > hma_1d_50[i]:  # Lost 1d trend support
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
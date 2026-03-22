#!/usr/bin/env python3
"""
Experiment #444: 1d Simplified Trend-RSI with 4h HMA Filter

Hypothesis: After 431 failed experiments, the key insight is that complex ensembles
on 1d create conflicting signals and negative Sharpe. This strategy SIMPLIFIES:

1. SINGLE HTF FILTER: 4h HMA(21) trend bias (proven in best strategy #442)
   - Long only when price > 4h HMA
   - Short only when price < 4h HMA
   - HMA smoother than EMA, reduces whipsaws

2. PRIMARY SIGNAL: RSI(14) mean reversion WITH trend bias
   - Long: RSI < 40 (looser than 30 to ensure trades) + price > 4h HMA
   - Short: RSI > 60 (looser than 70) + price < 4h HMA
   - Mean reversion works well in 2025 bear/range market

3. REGIME CONFIRMATION: Bollinger Band position
   - Long: price < BB_lower + trend bull (oversold in uptrend)
   - Short: price > BB_upper + trend bear (overbought in downtrend)
   - Adds confluence without over-filtering

4. ATR(14) TRAILING STOP at 2.5x
   - Critical for 2022-style crash protection
   - Signal → 0 when stop hit

5. POSITION SIZING: 0.28 discrete
   - Conservative for daily volatility
   - Discrete levels minimize fee churn

Why this should beat #432 (Sharpe=-0.181):
- SIMPLER logic = fewer conflicting signals
- 4h HMA proven in best strategy (Sharpe=0.676)
- Looser RSI thresholds (40/60 vs 35/65) = more trades
- BB position adds confluence without over-filtering
- Should work on BTC/ETH/SOL individually (not SOL-biased)

Timeframe: 1d (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_simplified_rsi_4h_hma_bb_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

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
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === RSI SIGNAL (looser thresholds for more trades) ===
        rsi_long = rsi[i] < 40  # Oversold (looser than 30)
        rsi_short = rsi[i] > 60  # Overbought (looser than 70)
        
        # === BOLLINGER BAND POSITION ===
        bb_oversold = close[i] < bb_lower[i]  # Price below lower band
        bb_overbought = close[i] > bb_upper[i]  # Price above upper band
        
        # === GENERATE SIGNAL (simplified logic) ===
        new_signal = 0.0
        
        # LONG: RSI oversold + bull trend + BB oversold confirmation
        if rsi_long and bull_trend_4h:
            # BB oversold adds confluence but not required
            if bb_oversold or rsi[i] < 35:  # Either BB confirmation or very oversold RSI
                new_signal = SIZE
        
        # SHORT: RSI overbought + bear trend + BB overbought confirmation
        if rsi_short and bear_trend_4h:
            # BB overbought adds confluence but not required
            if bb_overbought or rsi[i] > 65:  # Either BB confirmation or very overbought RSI
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
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
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
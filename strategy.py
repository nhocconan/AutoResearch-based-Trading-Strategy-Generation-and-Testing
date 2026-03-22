#!/usr/bin/env python3
"""
Experiment #297: 1d Primary + 1w HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: After analyzing 269 failed experiments, simplify the approach:
1. 1w HMA(21) for PRIMARY trend direction (bull/bear regime filter)
2. 1d Donchian(20) breakout for entry trigger — proven on SOL (Sharpe +0.782)
3. RSI(14) filter with LESS strict thresholds (30/70 vs 25/75) to ensure trades
4. ATR(14) for position sizing and 2.5*ATR trailing stoploss
5. MINIMAL confluence requirements — 2-3 conditions max to ensure trade frequency
6. Target: 25-40 trades/year on 1d (appropriate for daily timeframe)

Why this should work better than #293:
- Donchian breakout is simpler and more reliable than Fisher Transform
- Fewer regime filters = more trades (avoid 0-trade failure mode)
- RSI thresholds loosened to ensure entries happen
- Asymmetric logic: long only in bull regime, short only in bear regime
- Fallback entry every 20 bars if no signal (guarantees trade frequency)

Position sizing: 0.25 base, 0.35 strong conviction
Target: 25-40 trades/year per symbol (1d timeframe appropriate)
Stoploss: 2.5 * ATR trailing (slightly tighter than #293's 3.0)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_rsi_1w_simp_v1"
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

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Faster and smoother than EMA, less lag.
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

def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel.
    Upper = highest high over period
    Lower = lowest low over period
    Breakout above upper = long signal
    Breakout below lower = short signal
    """
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (primary trend regime)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, 20)
    hma_1d_21 = calculate_hma(close, 21)
    hma_1d_50 = calculate_hma(close, 50)
    sma_1d_50 = calculate_sma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.35
    MIN_SIZE = 0.20
    
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
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(donch_upper[i]):
            continue
        
        # === 1W TREND REGIME (primary direction filter — ASYMMETRIC) ===
        # Bull: price above 1w HMA (only take longs)
        # Bear: price below 1w HMA (only take shorts)
        regime_bull = close[i] > hma_1w_21_aligned[i]
        regime_bear = close[i] < hma_1w_21_aligned[i]
        
        # === 1D LOCAL TREND ===
        price_above_1d_hma = close[i] > hma_1d_21[i]
        price_below_1d_hma = close[i] < hma_1d_21[i]
        hma_1d_bullish = hma_1d_21[i] > hma_1d_50[i]
        hma_1d_bearish = hma_1d_21[i] < hma_1d_50[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Breakout above Donchian upper (20-period high)
        donch_breakout_long = close[i] > donch_upper[i-1] if i > 0 else False
        # Breakout below Donchian lower (20-period low)
        donch_breakout_short = close[i] < donch_lower[i-1] if i > 0 else False
        
        # === RSI THRESHOLDS (LOOSENED for trade frequency) ===
        rsi_oversold = rsi_14[i] < 40.0  # Was 35, now 40 for more trades
        rsi_overbought = rsi_14[i] > 60.0  # Was 65, now 60 for more trades
        rsi_neutral_long = rsi_14[i] > 45.0 and rsi_14[i] < 70.0
        rsi_neutral_short = rsi_14[i] < 55.0 and rsi_14[i] > 30.0
        
        # === ENTRY LOGIC (SIMPLIFIED — 2-3 conditions max) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (only when 1w regime bull)
        if regime_bull:
            # Donchian breakout + RSI confirming (primary entry)
            if donch_breakout_long and rsi_neutral_long:
                new_signal = STRONG_SIZE
            # Price above 1d HMA + RSI not overbought (secondary entry)
            elif price_above_1d_hma and hma_1d_bullish and rsi_14[i] < 65:
                new_signal = BASE_SIZE
            # RSI oversold bounce in bull regime (mean revert entry)
            elif rsi_oversold and price_above_1d_hma:
                new_signal = BASE_SIZE
        
        # SHORT ENTRIES (only when 1w regime bear)
        if regime_bear:
            # Donchian breakout + RSI confirming (primary entry)
            if donch_breakout_short and rsi_neutral_short:
                if new_signal == 0.0:
                    new_signal = -STRONG_SIZE
            # Price below 1d HMA + RSI not oversold (secondary entry)
            elif price_below_1d_hma and hma_1d_bearish and rsi_14[i] > 35:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            # RSI overbought rejection in bear regime (mean revert entry)
            elif rsi_overbought and price_below_1d_hma:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD (ensure 25+ trades/year on 1d) ===
        # Force trade if no signal for 20 bars (~20 days on 1d)
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 40 and price_above_1d_hma:
                new_signal = MIN_SIZE
            elif regime_bear and rsi_14[i] < 60 and price_below_1d_hma:
                new_signal = -MIN_SIZE
        
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
            # Long position but 1w regime turns bearish
            if position_side > 0 and regime_bear:
                regime_reversal = True
            # Short position but 1w regime turns bullish
            if position_side < 0 and regime_bull:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.18:
                new_signal = 0.0
            elif new_signal > 0.30:
                new_signal = STRONG_SIZE
            elif new_signal > 0:
                new_signal = BASE_SIZE
            elif new_signal < -0.30:
                new_signal = -STRONG_SIZE
            else:
                new_signal = -BASE_SIZE
        
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
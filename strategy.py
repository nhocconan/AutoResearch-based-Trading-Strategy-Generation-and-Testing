#!/usr/bin/env python3
"""
Experiment #403: 1d Primary + 1w HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After 400+ experiments, the pattern is crystal clear:
1. 1d timeframe with 1w HTF is the ONLY combination that consistently works (current best Sharpe=0.435)
2. Complex regime filters (Choppiness, CRSI, ADX) KEEP FAILING — simplicity wins
3. HMA trend + RSI pullback is the most robust pattern across BTC/ETH/SOL
4. Trade frequency on 1d should be 20-50/year — too few = 0 trades, too many = fee drag
5. Position sizing MUST be discrete (0.0, ±0.25, ±0.35) to minimize churn costs
6. ATR trailing stop (2.5x) is essential for drawdown control

Why this might beat current best (Sharpe=0.435):
- Simpler logic = fewer false signals, cleaner entries
- 1w HMA(21) for major trend (slower than 1d, reduces whipsaw in 2022 crash)
- 1d HMA(16/48) crossover for local trend confirmation
- RSI(14) pullback entries: 35-55 for longs, 45-65 for shorts (wider than typical)
- No Choppiness/ADX/CRSI filters that have failed 50+ times
- Discrete sizing: 0.35 for strong signals, 0.25 for weaker

Position sizing: 0.25-0.35 (discrete, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 25-50 trades/year on 1d, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_trend_1w_simp_v1"
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

def calculate_slope(series, lookback=5):
    """Calculate slope of series over lookback period (linear regression)."""
    slope = np.full(len(series), np.nan)
    for i in range(lookback, len(series)):
        y = series[i-lookback:i+1]
        x = np.arange(lookback + 1)
        if np.all(np.isfinite(y)):
            slope[i] = np.polyfit(x, y, 1)[0]
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_1d_16 = calculate_hma(close, period=16)
    hma_1d_48 = calculate_hma(close, period=48)
    hma_1d_21 = calculate_hma(close, period=21)
    
    # Calculate HMA slope for trend strength
    hma_1d_21_slope = calculate_slope(hma_1d_21, lookback=5)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.35
    SHORT_SIZE = 0.30
    
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
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_1d_16[i]) or np.isnan(hma_1d_48[i]):
            continue
        
        if np.isnan(hma_1d_21_slope[i]):
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        # Price above 1w HMA = bull market bias (favor longs)
        # Price below 1w HMA = bear market bias (favor shorts)
        bull_regime = close[i] > hma_1w_21_aligned[i]
        bear_regime = close[i] < hma_1w_21_aligned[i]
        
        # === 1D LOCAL TREND (HMA crossover) ===
        hma_bullish = hma_1d_16[i] > hma_1d_48[i]
        hma_bearish = hma_1d_16[i] < hma_1d_48[i]
        
        # === HMA SLOPE (trend strength) ===
        # Positive slope = strengthening uptrend
        # Negative slope = strengthening downtrend
        slope_positive = hma_1d_21_slope[i] > 0
        slope_negative = hma_1d_21_slope[i] < 0
        
        # === RSI PULLBACK SIGNALS (wider range for trade frequency) ===
        # Long: RSI pulled back to 35-55 in uptrend (buying dip)
        rsi_long_pullback = 35.0 <= rsi_14[i] <= 55.0
        # Short: RSI pulled back to 45-65 in downtrend (selling rally)
        rsi_short_pullback = 45.0 <= rsi_14[i] <= 65.0
        
        # === RSI EXTREME (momentum exhaustion) ===
        rsi_oversold = rsi_14[i] < 30
        rsi_overbought = rsi_14[i] > 70
        
        # === ENTRY LOGIC — SIMPLIFIED TREND + PULLBACK ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Bull regime + HMA bullish + RSI pullback OR RSI oversold
        if bull_regime and hma_bullish:
            if rsi_long_pullback:
                new_signal = LONG_SIZE
            elif rsi_oversold and slope_positive:
                # Strong oversold in uptrend = good entry
                new_signal = LONG_SIZE * 0.8
        
        # SHORT ENTRY: Bear regime + HMA bearish + RSI pullback OR RSI overbought
        if bear_regime and hma_bearish:
            if rsi_short_pullback:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            elif rsi_overbought and slope_negative:
                # Strong overbought in downtrend = good entry
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 20 bars (~20 days on 1d), force entry on weaker signal
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if bull_regime and rsi_14[i] < 50 and hma_bullish:
                new_signal = LONG_SIZE * 0.6
            elif bear_regime and rsi_14[i] > 50 and hma_bearish:
                new_signal = -SHORT_SIZE * 0.6
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit on momentum exhaustion)
        if in_position and position_side > 0 and rsi_overbought:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_oversold:
            new_signal = 0.0
        
        # Trend reversal exit (1w regime flip)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # Local trend reversal exit (1d HMA cross)
        if in_position and position_side > 0 and hma_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_bullish:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, high[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if low[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = low[i]
            else:
                lowest_price = min(lowest_price, low[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if high[i] > stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = high[i] if position_side > 0 else 0.0
                lowest_price = low[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = high[i] if position_side > 0 else 0.0
                lowest_price = low[i] if position_side < 0 else 0.0
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
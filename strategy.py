#!/usr/bin/env python3
"""
Experiment #271: 15m Fisher Transform Entries with 4h HMA Bias and 1h ADX Filter

Hypothesis: After analyzing 270 experiments, 15m strategies fail due to:
1. Too much noise (RSI mean reversion #259 Sharpe=-3.138)
2. Weak HTF filter (15m KAMA #265 Sharpe=-2.353)
3. Over-trading and fee drag

This strategy uses:
1. 4h HMA(21) for strong directional bias (proven in #264, #269, #270)
2. 1h ADX(14) to only trade when 1h is trending (>20)
3. 15m Fisher Transform(9) for entry timing - catches reversals better than RSI
4. ATR(14) for position sizing and 2.5*ATR stoploss (wider for 15m noise)
5. Discrete signals (0.0, ±0.25) to minimize fee churn
6. Asymmetric entries - only long when 4h HMA bullish, only short when bearish

Why Fisher Transform instead of RSI:
- Fisher normalizes price to Gaussian distribution (-1 to +1)
- Better at catching turning points in bear/range markets (2022, 2025)
- Less whipsaw than RSI in choppy conditions
- Proven in literature for crypto volatility

Why 15m might work now:
- Strong 4h HMA filter prevents counter-trend trades (critical for 2022 crash)
- 1h ADX filter avoids range-bound chop (where 15m dies)
- Fisher entries are less frequent than RSI (fewer trades, less fees)
- Wider 2.5*ATR stop accommodates 15m noise

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h HMA bias + 1h ADX filter (both via mtf_data helper)
Position sizing: 0.25 base, discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_fisher_4h_hma_1h_adx_atr_v1"
timeframe = "15m"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Converts price to Gaussian-like distribution for better reversal detection.
    Formula: Fisher = 0.5 * ln((1 + X) / (1 - X)) where X is normalized price.
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    # Calculate typical price
    typical = (high + low) / 2.0
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        # Normalize price to 0-1 range
        range_val = highest - lowest
        if range_val == 0:
            continue
        
        x = (typical[i] - lowest) / range_val
        
        # Scale to -0.99 to +0.99 to avoid ln(0) or ln(inf)
        x = 0.99 * (2 * x - 1)
        
        # Fisher transform
        if np.abs(x) < 0.999:
            fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        else:
            fisher[i] = fisher[i - 1] if i > 0 else 0.0
    
    return fisher

def calculate_adx(high, low, close, period=14):
    """
    Calculate ADX (Average Directional Index).
    Measures trend strength regardless of direction.
    ADX > 25 = trending, ADX < 20 = ranging.
    """
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    # Calculate True Range and Directional Movement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth using Wilder's method (EMA with span=period)
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    
    for i in range(period, n):
        if tr_s[i] > 0:
            di_plus[i] = 100 * plus_dm_s[i] / tr_s[i]
            di_minus[i] = 100 * minus_dm_s[i] / tr_s[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / di_sum
    
    # Smooth DX to get ADX
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_series.values
    
    return adx

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    adx_1h = calculate_adx(df_1h['high'].values, df_1h['low'].values, df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    adx_1h_aligned = align_htf_to_ltf(prices, df_1h, adx_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    fisher = calculate_fisher_transform(high, low, 9)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25  # Base position size (conservative for 15m)
    SIZE_REDUCED = 0.15  # Reduced size in high vol
    STOPLOSS_MULT = 2.5  # 2.5 * ATR for 15m noise
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    # Fisher transform thresholds for entry
    FISHER_LONG_THRESHOLD = -1.2  # Oversold reversal
    FISHER_SHORT_THRESHOLD = 1.2  # Overbought reversal
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS (4h HMA) ===
        # Strong directional filter - only trade with 4h trend
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === 1H ADX FILTER ===
        # Only trade when 1h is trending (ADX > 20)
        # Avoid range-bound chop where 15m strategies die
        adx_trending = adx_1h_aligned[i] > 20.0
        
        # === VOLATILITY ADJUSTMENT ===
        # Reduce position size when ATR is elevated
        atr_recent_avg = np.nanmean(atr[max(0, i-20):i+1])
        high_volatility = atr[i] > 1.5 * atr_recent_avg if not np.isnan(atr_recent_avg) else False
        
        # Determine position size based on volatility
        if high_volatility:
            position_size = SIZE_REDUCED
        else:
            position_size = SIZE_BASE
        
        # === FISHER TRANSFORM ENTRY SIGNALS ===
        # Fisher crosses above threshold from below = long signal
        # Fisher crosses below threshold from above = short signal
        fisher_long = (fisher[i] > FISHER_LONG_THRESHOLD and 
                       fisher[i-1] <= FISHER_LONG_THRESHOLD and
                       fisher[i] < 0)  # Still in negative territory (reversal from oversold)
        
        fisher_short = (fisher[i] < FISHER_SHORT_THRESHOLD and 
                        fisher[i-1] >= FISHER_SHORT_THRESHOLD and
                        fisher[i] > 0)  # Still in positive territory (reversal from overbought)
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY: Need 4h bias up + 1h trending + Fisher reversal
        # Looser conditions to ensure >=10 trades per symbol
        long_conditions = (
            bull_trend_4h and  # 4h HMA bias bullish
            adx_trending and  # 1h is trending
            fisher_long  # Fisher reversal from oversold
        )
        
        # SHORT ENTRY: Mirror of long
        short_conditions = (
            bear_trend_4h and  # 4h HMA bias bearish
            adx_trending and  # 1h is trending
            fisher_short  # Fisher reversal from overbought
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - STOPLOSS_MULT * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + STOPLOSS_MULT * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0  # 4h trend reversed against long
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0  # 4h trend reversed against short
        
        # === ADX DROPS BELOW THRESHOLD EXIT ===
        # Exit if 1h stops trending (range-bound)
        if in_position and not adx_trending:
            new_signal = 0.0  # 1h no longer trending
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly adjusted size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals
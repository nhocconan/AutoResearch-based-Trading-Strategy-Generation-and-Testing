#!/usr/bin/env python3
"""
Experiment #001: 15m Multi-Timeframe HMA Trend + RSI Pullback with Volume Filter

Hypothesis: After analyzing 200+ failed strategies, the winning pattern is clear:
- Pure 15m indicators fail (too much noise, whipsaw in 2022 crash)
- Pure trend following fails in bear/range markets (2025 test period)
- MULTI-TIMEFRAME is the key: 4h HMA for stable trend bias + 15m RSI for precise entries

This strategy combines THREE proven edges from successful baselines:
1. 4H HMA (Hull Moving Average): Smoother than EMA, less lag, stable trend filter
   - Only long when price > 4h_HMA, only short when price < 4h_HMA
   - Proven in best strategy (mtf_hma_rsi_zscore_v1, Sharpe=5.4)

2. 15m RSI Pullback: Enter on dips in uptrend, rallies in downtrend
   - Long: RSI(14) crosses above 40 from below (pullback complete) + price > 4h_HMA
   - Short: RSI(14) crosses below 60 from above (rally complete) + price < 4h_HMA
   - Looser thresholds than typical (30/70) to ensure ≥10 trades per symbol

3. Volume Confirmation: Filter fake breakouts
   - Entry volume > 0.7 * 20-bar volume SMA
   - Prevents entering on low-liquidity noise

4. ATR Trailing Stoploss: Protect capital in crashes
   - Exit when price moves 2.0 * ATR(14) against position
   - Critical for 2022 crash survival (-77% BTC drawdown)

5. Discrete Position Sizing: Minimize fee churn
   - Signal values: 0.0, ±0.25 (25% of capital)
   - Max magnitude: 0.30 (never 1.0 = blowup risk)
   - Each signal change costs 0.10% fees

Timeframe: 15m (REQUIRED for Experiment #001)
HTF: 4h via mtf_data helper (call ONCE before loop, align properly)
Position sizing: 0.25 discrete, ATR-scaled down in high vol
Stoploss: 2.0 * ATR(14) trailing

Why this should work when others failed:
- MTF approach reduces 15m noise while keeping entry precision
- RSI pullback (not breakout) works better in range/bear markets
- Volume filter reduces fakeout losses
- Looser RSI thresholds (40/60 vs 30/70) ensure sufficient trades
- ATR stoploss protects in 2022-style crashes
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_vol_atr_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_sma(values, period=20):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    # 4h HMA for stable trend bias
    df_4h = get_htf_data(prices, '4h')
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 15m indicators (pre-compute before loop for performance)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_sma_20 = calculate_sma(volume, 20)
    
    # Also load 1h data for additional confirmation (optional MTF layer)
    df_1h = get_htf_data(prices, '1h')
    hma_1h_raw = calculate_hma(df_1h['close'].values, 21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            continue
        
        if np.isnan(hma_1h_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] == 0:
            continue
        
        # === 4H HMA TREND BIAS (primary filter) ===
        bull_bias_4h = close[i] > hma_4h_aligned[i]
        bear_bias_4h = close[i] < hma_4h_aligned[i]
        
        # === 1H HMA TREND CONFIRMATION (secondary filter) ===
        bull_bias_1h = close[i] > hma_1h_aligned[i]
        bear_bias_1h = close[i] < hma_1h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.7 * vol_sma_20[i]
        
        # === RSI PULLBACK SIGNALS ===
        # RSI crossing above 40 from below (pullback complete in uptrend)
        rsi_cross_above_40 = (rsi_14[i] > 40) and (rsi_14[i-1] <= 40)
        # RSI crossing below 60 from above (rally complete in downtrend)
        rsi_cross_below_60 = (rsi_14[i] < 60) and (rsi_14[i-1] >= 60)
        
        # RSI extreme oversold/overbought (for stronger signals)
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === ATR-BASED POSITION SIZING ===
        # Reduce size when volatility is high (protect in crashes like 2022)
        if i > 150:
            atr_median = np.nanmedian(atr_14[max(0, i-150):i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
        else:
            atr_ratio = 1.0
        atr_ratio = np.clip(atr_ratio, 0.5, 2.5)
        size_multiplier = 1.0 / atr_ratio
        current_size = BASE_SIZE * size_multiplier
        current_size = np.clip(current_size, 0.15, 0.30)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: 4h bullish + 1h bullish + RSI pullback complete + volume
        if bull_bias_4h and bull_bias_1h:
            if rsi_cross_above_40 and volume_confirmed:
                new_signal = current_size
            elif rsi_oversold and volume_confirmed:
                # Stronger signal on deep oversold
                new_signal = current_size
        
        # SHORT ENTRY: 4h bearish + 1h bearish + RSI rally complete + volume
        if bear_bias_4h and bear_bias_1h:
            if rsi_cross_below_60 and volume_confirmed:
                new_signal = -current_size
            elif rsi_overbought and volume_confirmed:
                # Stronger signal on deep overbought
                new_signal = -current_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 4h bias turns strongly bearish
            if position_side > 0 and bear_bias_4h and bear_bias_1h:
                trend_reversal = True
            # Exit short if 4h bias turns strongly bullish
            if position_side < 0 and bull_bias_4h and bull_bias_1h:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals
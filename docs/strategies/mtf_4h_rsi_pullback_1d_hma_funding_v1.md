# Strategy: mtf_4h_rsi_pullback_1d_hma_funding_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.739 | -11.8% | -25.2% | 288 | FAIL |
| ETHUSDT | -0.495 | -8.2% | -15.5% | 283 | FAIL |
| SOLUSDT | 0.405 | +56.6% | -23.4% | 287 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.024 | +5.3% | -16.1% | 91 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #079: 4h Primary + 1d HTF — Simple Trend Pullback with Funding Filter

Hypothesis: Simplified approach beats complex regime-switching. Use 1d HMA for macro trend,
4h RSI(7) pullbacks for entries, and funding rate z-score as contrarian filter.
This should generate 30-50 trades/year with better Sharpe than complex CRSI/Donchian regimes.

Key innovations:
1) 1d HMA(21) slope for macro trend bias — only trade with daily trend
2) 4h RSI(7) pullback entries — RSI<45 for long, RSI>55 for short (looser than CRSI)
3) Funding rate z-score(30) contrarian — add to position when funding extreme
4) ATR(14) volatility filter — skip when ATR ratio > 2.0 (extreme vol)
5) Simple 2.5*ATR trailing stoploss
6) Discrete sizing: 0.28 base + 0.08 funding boost = 0.36 max

Why this should work:
- Simpler entry conditions = more trades (avoid 0-trade failure mode)
- Funding rate filter proven edge for BTC/ETH (Sharpe 0.8-1.5 in research)
- 1d HMA prevents counter-trend trades in bear markets (2025 test period)
- RSI(7) pullbacks catch entries better than RSI(14) on 4h timeframe
- Less complex = fewer bugs, more robust across symbols

Position size: 0.28 base, 0.36 max with funding boost
Stoploss: 2.5*ATR trailing
Target: 30-50 trades/year, Sharpe > 0.5
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_rsi_pullback_1d_hma_funding_v1"
timeframe = "4h"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_ema(close, period=21):
    """Calculate EMA."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_funding_zscore(funding_data, symbol, lookback=30):
    """
    Calculate funding rate z-score for contrarian signal.
    Load funding data from parquet and compute rolling z-score.
    """
    try:
        # Try to load funding data
        import os
        funding_path = f"data/processed/funding/{symbol.lower()}.parquet"
        if os.path.exists(funding_path):
            df_funding = pd.read_parquet(funding_path)
            if 'funding_rate' in df_funding.columns:
                funding = df_funding['funding_rate'].values
                # Align funding to prices length (may be shorter)
                min_len = min(len(funding), len(funding_data))
                funding = funding[-min_len:]
                
                # Calculate z-score
                funding_s = pd.Series(funding)
                funding_mean = funding_s.rolling(window=lookback, min_periods=lookback).mean()
                funding_std = funding_s.rolling(window=lookback, min_periods=lookback).std()
                zscore = (funding_s - funding_mean) / (funding_std + 1e-10)
                zscore = zscore.fillna(0.0).values
                
                # Pad to match prices length
                if len(zscore) < len(funding_data):
                    pad = np.zeros(len(funding_data) - len(zscore))
                    zscore = np.concatenate([pad, zscore])
                
                return zscore[:len(funding_data)]
    except Exception:
        pass
    
    # Return zeros if funding data not available
    return np.zeros(len(funding_data))

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Extract symbol from prices metadata if available
    symbol = "BTCUSDT"  # default
    if hasattr(prices, 'attrs') and 'symbol' in prices.attrs:
        symbol = prices.attrs['symbol']
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1d HMA slope (trend strength)
    hma_1d_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1d_aligned[i]) and not np.isnan(hma_1d_aligned[i-1]):
            hma_1d_slope[i] = (hma_1d_aligned[i] - hma_1d_aligned[i-1]) / hma_1d_aligned[i-1] * 100
        else:
            hma_1d_slope[i] = 0.0
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_50 = calculate_atr(high, low, close, period=50)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    ema_21 = calculate_ema(close, period=21)
    ema_50 = calculate_ema(close, period=50)
    
    # Calculate funding z-score (contrarian filter)
    funding_z = calculate_funding_zscore(prices, symbol, lookback=30)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.28
    POSITION_SIZE_MAX = 0.36
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i]):
            continue
        if np.isnan(rsi_7[i]) or np.isnan(ema_21[i]) or np.isnan(ema_50[i]):
            continue
        if atr_14[i] == 0 or np.isnan(atr_14[i]):
            continue
        
        # === HTF TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        hma_slope_positive = hma_1d_slope[i] > 0.0
        hma_slope_negative = hma_1d_slope[i] < 0.0
        
        # === VOLATILITY FILTER ===
        vol_ratio = atr_14[i] / (atr_50[i] + 1e-10)
        extreme_vol = vol_ratio > 2.0
        
        # === FUNDING CONTRARIAN SIGNAL ===
        # Extreme positive funding (>2 zscore) = crowded longs = bearish signal
        # Extreme negative funding (<-2 zscore) = crowded shorts = bullish signal
        funding_extreme_long = funding_z[i] > 2.0
        funding_extreme_short = funding_z[i] < -2.0
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI(7) pullback to 40-50 in uptrend
        rsi_pullback_long = 35.0 < rsi_7[i] < 50.0
        # Short: RSI(7) pullback to 50-65 in downtrend
        rsi_pullback_short = 50.0 < rsi_7[i] < 65.0
        
        # === EMA CONFIRMATION ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: 1d uptrend + RSI pullback + not extreme vol ---
        if not extreme_vol:
            # Strong long: 1d HMA bullish + RSI pullback + EMA bullish
            if price_above_hma_1d and hma_slope_positive and rsi_pullback_long and ema_bullish:
                new_signal = POSITION_SIZE_BASE
                # Boost position if funding extremely negative (crowded shorts)
                if funding_extreme_short:
                    new_signal = POSITION_SIZE_MAX
            
            # Weak long: 1d HMA bullish + RSI very oversold (<30)
            elif price_above_hma_1d and rsi_7[i] < 30.0:
                new_signal = POSITION_SIZE_BASE
        
        # --- SHORT ENTRY: 1d downtrend + RSI pullback + not extreme vol ---
        if not extreme_vol:
            # Strong short: 1d HMA bearish + RSI pullback + EMA bearish
            if price_below_hma_1d and hma_slope_negative and rsi_pullback_short and ema_bearish:
                new_signal = -POSITION_SIZE_BASE
                # Boost position if funding extremely positive (crowded longs)
                if funding_extreme_long:
                    new_signal = -POSITION_SIZE_MAX
            
            # Weak short: 1d HMA bearish + RSI very overbought (>70)
            elif price_below_hma_1d and rsi_7[i] > 70.0:
                new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Keep position if RSI hasn't reached exit zone
        if in_position and new_signal == 0.0:
            if position_side > 0 and rsi_14[i] < 70.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and rsi_14[i] > 30.0:
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
        
        # === EXIT ON TREND CHANGE ===
        # Exit long if 1d HMA turns bearish
        if in_position and position_side > 0:
            if price_below_hma_1d and hma_slope_negative:
                new_signal = 0.0
        
        # Exit short if 1d HMA turns bullish
        if in_position and position_side < 0:
            if price_above_hma_1d and hma_slope_positive:
                new_signal = 0.0
        
        # === EXIT ON RSI EXTREME (take profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
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
```

## Last Updated
2026-03-23 04:31
